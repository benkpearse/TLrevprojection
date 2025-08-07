import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# --- Page Configuration ---
st.set_page_config(
    page_title="A/B Test Revenue & ABV Projection",
    page_icon="�",
    layout="wide"
)

# --- Initialize Session State ---
# This is crucial for storing results after a button click
if 'results_calculated' not in st.session_state:
    st.session_state.results_calculated = False

# --- App Title and Description ---
col_logo1, col_logo2 = st.columns([1, 5])
with col_logo1:
    st.image("https://placehold.co/150x150/003580/FFFFFF?text=Logo", width=100)
with col_logo2:
    st.title("A/B Test Revenue & ABV Projection Tool")
    st.markdown("A tool to project revenue and evaluate A/B tests that affect **Conversion Rate** and/or **Average Booking Value (ABV)**.")

st.divider()

# --- Input Section ---
st.sidebar.header("⚙️ 1. A/B Test Results")
baseline_conv = st.sidebar.number_input(
    "Control Conversion Rate (%)", 
    min_value=0.0, max_value=100.0, value=2.5, step=0.1,
    help="The conversion rate of the original version (Control)."
)
variant_conv = st.sidebar.number_input(
    "Variant Conversion Rate (%)", 
    min_value=0.0, max_value=100.0, value=2.5, step=0.1,
    help="The conversion rate of the new version (Variant)."
)
baseline_users = st.sidebar.number_input(
    "Control Sample Size (Visitors)", 
    min_value=1, value=10000, step=100,
    help="Number of unique visitors who saw the Control version."
)
variant_users = st.sidebar.number_input(
    "Variant Sample Size (Visitors)", 
    min_value=1, value=10000, step=100,
    help="Number of unique visitors who saw the Variant version."
)

st.sidebar.header("💰 2. Revenue & Forecast Settings")
baseline_avg_revenue = st.sidebar.number_input(
    "Control Average Booking Value ($)", 
    min_value=0.0, value=120.0, step=5.0,
    help="The average revenue per conversion for the Control group."
)
variant_avg_revenue = st.sidebar.number_input(
    "Variant Average Booking Value ($)", 
    min_value=0.0, value=130.0, step=5.0,
    help="The average revenue per conversion for the Variant group. Change this to model tests on price or add-ons."
)
daily_traffic = st.sidebar.number_input(
    "Projected Average Daily Visitors", 
    min_value=1, value=5000, step=100,
    help="The expected number of unique visitors to the page each day going forward."
)
forecast_period = st.sidebar.number_input(
    "Forecast Period (days)", 
    min_value=1, value=90, step=1,
    help="How many days into the future to project revenue."
)

# Optional Decay Settings
use_decay = st.sidebar.checkbox("Account for Uplift Decay?")
decay_rate_pct = 0
if use_decay:
    decay_rate_pct = st.sidebar.slider(
        "Uplift Decay Rate (%)",
        min_value=0, max_value=100, value=10, step=5,
        help="The percentage by which the initial total uplift is expected to decrease by the end of the forecast period."
    )

st.sidebar.divider()

# --- Calculation Trigger ---
if st.sidebar.button("🚀 Run Calculation", type="primary"):
    st.session_state.results_calculated = True
    
    # --- Calculations ---
    # Convert percentages to rates
    baseline_rate = baseline_conv / 100
    variant_rate = variant_conv / 100
    decay_rate = decay_rate_pct / 100

    # Calculate number of conversions from the test for significance testing
    baseline_conversions_test = baseline_rate * baseline_users
    variant_conversions_test = variant_rate * variant_users

    # --- Uplift Calculations ---
    cvr_abs_uplift = variant_rate - baseline_rate
    cvr_rel_uplift_pct = (cvr_abs_uplift / baseline_rate) * 100 if baseline_rate > 0 else np.nan
    abv_abs_uplift = variant_avg_revenue - baseline_avg_revenue
    abv_rel_uplift_pct = (abv_abs_uplift / baseline_avg_revenue) * 100 if baseline_avg_revenue > 0 else np.nan

    # --- Statistical Significance (Z-test for two proportions) ---
    p_pool = (baseline_conversions_test + variant_conversions_test) / (baseline_users + variant_users)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / baseline_users + 1 / variant_users))
    z_score = cvr_abs_uplift / se_pool
    p_value = stats.norm.sf(abs(z_score)) * 2

    # --- Projection Calculation ---
    dates = pd.date_range(start=pd.Timestamp.today(), periods=forecast_period)
    daily_baseline_revenue = daily_traffic * baseline_rate * baseline_avg_revenue
    daily_variant_initial_revenue = daily_traffic * variant_rate * variant_avg_revenue
    initial_daily_lift = daily_variant_initial_revenue - daily_baseline_revenue

    daily_variant_revenues = []
    for day in range(forecast_period):
        daily_decay_factor = (day / (forecast_period - 1)) if forecast_period > 1 else 0
        decayed_lift = initial_daily_lift * (1 - decay_rate * daily_decay_factor)
        daily_variant_revenue = daily_baseline_revenue + decayed_lift
        daily_variant_revenues.append(daily_variant_revenue)

    control_cumulative = np.cumsum([daily_baseline_revenue] * forecast_period)
    variant_cumulative = np.cumsum(daily_variant_revenues)

    projection_df = pd.DataFrame({
        'Control Cumulative Revenue': control_cumulative,
        'Variant Cumulative Revenue': variant_cumulative,
    }, index=dates)

    # --- Store results in session state ---
    st.session_state.p_value = p_value
    st.session_state.cvr_abs_uplift = cvr_abs_uplift
    st.session_state.cvr_rel_uplift_pct = cvr_rel_uplift_pct
    st.session_state.abv_abs_uplift = abv_abs_uplift
    st.session_state.abv_rel_uplift_pct = abv_rel_uplift_pct
    st.session_state.proj_baseline_revenue = control_cumulative[-1]
    st.session_state.proj_variant_revenue = variant_cumulative[-1]
    st.session_state.proj_revenue_diff = variant_cumulative[-1] - control_cumulative[-1]
    st.session_state.projection_df = projection_df
    st.session_state.decay_rate_pct = decay_rate_pct
    st.session_state.forecast_period = forecast_period

# --- Results Display Area ---
if not st.session_state.results_calculated:
    st.info("Please enter your A/B test data in the sidebar and click 'Run Calculation'.")
else:
    st.header("📊 Key Results Summary")

    # Significance interpretation
    significance_level = 0.05
    st.subheader("Conversion Rate Significance")
    if st.session_state.p_value < significance_level and st.session_state.cvr_abs_uplift > 0:
        st.success(f"**Result is Statistically Significant** (p-value: {st.session_state.p_value:.4f})")
        st.markdown(f"We are confident the change in **conversion rate** is a real improvement.")
    else:
        st.warning(f"**Result is Not Statistically Significant** (p-value: {st.session_state.p_value:.4f})")
        st.markdown(f"We cannot be confident the change in **conversion rate** is real. Projections should be treated with caution.")

    st.info("Note: Statistical significance for Average Booking Value requires a different test (e.g., t-test) which is not included here.")

    # Metrics columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Conversion Rate Uplift", f"{st.session_state.cvr_rel_uplift_pct:.2f}%", delta=f"{st.session_state.cvr_abs_uplift*100:.2f} p.p.")
    col2.metric("ABV Uplift", f"{st.session_state.abv_rel_uplift_pct:.2f}%", delta=f"${st.session_state.abv_abs_uplift:.2f}")
    col3.metric("P-Value (for CVR)", f"{st.session_state.p_value:.4f}")

    st.divider()

    # --- Revenue Projection Section ---
    st.header(f"💰 Revenue Projection Over {st.session_state.forecast_period} Days")
    st.markdown(f"Based on an average of **{daily_traffic:,} visitors per day** and a **{st.session_state.decay_rate_pct}% uplift decay**.")
    
    col_rev1, col_rev2, col_rev3 = st.columns(3)
    col_rev1.metric("Projected Control Revenue", f"${st.session_state.proj_baseline_revenue:,.0f}")
    col_rev2.metric("Projected Variant Revenue", f"${st.session_state.proj_variant_revenue:,.0f}")
    col_rev3.metric("Projected Revenue Lift", f"${st.session_state.proj_revenue_diff:,.0f}")

    st.subheader("Cumulative Revenue Projection")
    st.line_chart(st.session_state.projection_df)
    
    st.info(f"The Variant curve shows the combined impact of changes in CVR and ABV. It flattens slightly over time if the **{st.session_state.decay_rate_pct}%** decay assumption is used.")
�
