import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# --- Page Configuration ---
st.set_page_config(
    page_title="A/B Test Revenue & ABV Projection",
    page_icon="📈",
    layout="wide"
)

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
decay_rate_pct = st.sidebar.slider(
    "Uplift Decay Rate (%)",
    min_value=0, max_value=100, value=10, step=5,
    help="The percentage by which the initial total uplift (from CVR and ABV) is expected to decrease by the end of the forecast period."
)

# --- Calculations ---
# Convert percentages to rates
baseline_rate = baseline_conv / 100
variant_rate = variant_conv / 100
decay_rate = decay_rate_pct / 100

# Calculate number of conversions from the test for significance testing
baseline_conversions_test = baseline_rate * baseline_users
variant_conversions_test = variant_rate * variant_users

# --- Uplift Calculations ---
# Conversion Rate Uplift
cvr_abs_uplift = variant_rate - baseline_rate
cvr_rel_uplift_pct = (cvr_abs_uplift / baseline_rate) * 100 if baseline_rate > 0 else np.nan

# Average Booking Value (ABV) Uplift
abv_abs_uplift = variant_avg_revenue - baseline_avg_revenue
abv_rel_uplift_pct = (abv_abs_uplift / baseline_avg_revenue) * 100 if baseline_avg_revenue > 0 else np.nan

# --- Statistical Significance (Z-test for two proportions) ---
# This test is ONLY for the conversion rate. A different test (like a t-test) would be needed for revenue, but is more complex to implement here.
p_pool = (baseline_conversions_test + variant_conversions_test) / (baseline_users + variant_users)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / baseline_users + 1 / variant_users))
z_score = cvr_abs_uplift / se_pool
p_value = stats.norm.sf(abs(z_score)) * 2

# --- Results Display ---
st.header("📊 Key Results Summary")

# Significance interpretation
significance_level = 0.05
st.subheader("Conversion Rate Significance")
if p_value < significance_level and cvr_abs_uplift > 0:
    st.success(f"**Result is Statistically Significant** (p-value: {p_value:.4f})")
    st.markdown(f"We are confident the change in **conversion rate** is a real improvement.")
else:
    st.warning(f"**Result is Not Statistically Significant** (p-value: {p_value:.4f})")
    st.markdown(f"We cannot be confident the change in **conversion rate** is real. Projections should be treated with caution.")

st.info("Note: Statistical significance for Average Booking Value requires a different test (e.g., t-test) which is not included here. This section only evaluates the significance of the change in the rate of booking.")

# Metrics columns
col1, col2, col3 = st.columns(3)
col1.metric(
    "Conversion Rate Uplift",
    f"{cvr_rel_uplift_pct:.2f}%",
    delta=f"{cvr_abs_uplift*100:.2f} p.p.",
    help="The percentage improvement in conversion rate."
)
col2.metric(
    "ABV Uplift",
    f"{abv_rel_uplift_pct:.2f}%",
    delta=f"${abv_abs_uplift:.2f}",
    help="The percentage improvement in Average Booking Value."
)
col3.metric(
    "P-Value (for CVR)",
    f"{p_value:.4f}",
    help="The probability of observing the CVR change by random chance."
)

st.divider()

# --- Revenue Projection Section ---
st.header(f"💰 Revenue Projection Over {forecast_period} Days")
st.markdown(f"Based on an average of **{daily_traffic:,} visitors per day** and a **{decay_rate_pct}% uplift decay**.")

# --- Visualization & Projection Calculation ---
dates = pd.date_range(start=pd.Timestamp.today(), periods=forecast_period)

# Calculate daily revenue for each group
daily_baseline_revenue = daily_traffic * baseline_rate * baseline_avg_revenue
daily_variant_initial_revenue = daily_traffic * variant_rate * variant_avg_revenue

# The initial lift is the total difference in daily revenue
initial_daily_lift = daily_variant_initial_revenue - daily_baseline_revenue

# Calculate daily revenue for variant, applying decay to the LIFT
daily_variant_revenues = []
for day in range(forecast_period):
    daily_decay_factor = (day / (forecast_period - 1)) if forecast_period > 1 else 0
    decayed_lift = initial_daily_lift * (1 - decay_rate * daily_decay_factor)
    daily_variant_revenue = daily_baseline_revenue + decayed_lift
    daily_variant_revenues.append(daily_variant_revenue)

# Create a DataFrame for cumulative revenue
control_cumulative = np.cumsum([daily_baseline_revenue] * forecast_period)
variant_cumulative = np.cumsum(daily_variant_revenues)

projection_df = pd.DataFrame({
    'Control Cumulative Revenue': control_cumulative,
    'Variant Cumulative Revenue': variant_cumulative,
}, index=dates)

# Final projected totals from the cumulative calculation
proj_baseline_revenue = control_cumulative[-1]
proj_variant_revenue = variant_cumulative[-1]
proj_revenue_diff = proj_variant_revenue - proj_baseline_revenue

# Projection metrics
col_rev1, col_rev2, col_rev3 = st.columns(3)
col_rev1.metric("Projected Control Revenue", f"${proj_baseline_revenue:,.0f}")
col_rev2.metric("Projected Variant Revenue", f"${proj_variant_revenue:,.0f}")
col_rev3.metric(
    f"Projected Revenue Lift", 
    f"${proj_revenue_diff:,.0f}",
    help=f"The total additional revenue expected from the variant over {forecast_period} days, accounting for decay."
)

st.subheader("Cumulative Revenue Projection")
st.line_chart(projection_df)

st.info(f"The Variant curve shows the combined impact of changes in CVR and ABV. It flattens slightly over time due to the **{decay_rate_pct}%** decay assumption.")
