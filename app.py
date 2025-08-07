import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# --- Page Configuration ---
st.set_page_config(
    page_title="A/B Test Revenue Projection",
    page_icon="📈",
    layout="wide"
)

# --- App Title and Description ---
# Using columns to add a logo or image for branding
col_logo1, col_logo2 = st.columns([1, 5])
with col_logo1:
    # You can host a Travelodge logo online and link to it here
    st.image("https://placehold.co/150x150/003580/FFFFFF?text=Logo", width=100)
with col_logo2:
    st.title("A/B Test Revenue Projection Tool")
    st.markdown("A tool for the Travelodge team to project revenue and evaluate A/B test results, with optional uplift decay.")

st.divider()

# --- Input Section ---
st.sidebar.header("⚙️ 1. A/B Test Results")
baseline_conv = st.sidebar.number_input(
    "Control Conversion Rate (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=2.5, 
    step=0.1,
    help="The conversion rate of the original version (Control)."
)
variant_conv = st.sidebar.number_input(
    "Variant Conversion Rate (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=2.8, 
    step=0.1,
    help="The conversion rate of the new version (Variant)."
)
baseline_users = st.sidebar.number_input(
    "Control Sample Size (Visitors)", 
    min_value=1, 
    value=10000, 
    step=100,
    help="Number of unique visitors who saw the Control version."
)
variant_users = st.sidebar.number_input(
    "Variant Sample Size (Visitors)", 
    min_value=1, 
    value=10000, 
    step=100,
    help="Number of unique visitors who saw the Variant version."
)

st.sidebar.header("💰 2. Revenue & Forecast Settings")
avg_revenue = st.sidebar.number_input(
    "Average Revenue per Conversion ($)", 
    min_value=0.0, 
    value=120.0, 
    step=5.0,
    help="Also known as Average Booking Value (ABV)."
)
daily_traffic = st.sidebar.number_input(
    "Projected Average Daily Visitors", 
    min_value=1, 
    value=5000, 
    step=100,
    help="The expected number of unique visitors to the page each day going forward."
)
forecast_period = st.sidebar.number_input(
    "Forecast Period (days)", 
    min_value=1, 
    value=90, 
    step=1,
    help="How many days into the future to project revenue."
)
decay_rate_pct = st.sidebar.slider(
    "Uplift Decay Rate (%)",
    min_value=0,
    max_value=100,
    value=10,
    step=5,
    help="The percentage by which the initial uplift is expected to decrease by the end of the forecast period. 0% means no decay."
)


# --- Calculations ---
# Convert percentages to rates
baseline_rate = baseline_conv / 100
variant_rate = variant_conv / 100
decay_rate = decay_rate_pct / 100

# Calculate number of conversions from the test
baseline_conversions_test = baseline_rate * baseline_users
variant_conversions_test = variant_rate * variant_users

# Calculate uplift
abs_uplift = variant_rate - baseline_rate
rel_uplift_pct = (abs_uplift / baseline_rate) * 100 if baseline_rate > 0 else np.nan

# Statistical Significance (Z-test for two proportions)
p_pool = (baseline_conversions_test + variant_conversions_test) / (baseline_users + variant_users)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / baseline_users + 1 / variant_users))
z_score = abs_uplift / se_pool
p_value = stats.norm.sf(abs(z_score)) * 2  # Ensure z_score is positive for sf

# Confidence Interval for the absolute uplift
se_diff = np.sqrt(baseline_rate * (1 - baseline_rate) / baseline_users + variant_rate * (1 - variant_rate) / variant_users)
ci_low = abs_uplift - stats.norm.ppf(0.975) * se_diff
ci_high = abs_uplift + stats.norm.ppf(0.975) * se_diff

# --- Results Display ---
st.header("📊 Key Results Summary")
st.markdown("This section summarizes the statistical significance and uplift from the A/B test.")

# Significance interpretation
significance_level = 0.05
if p_value < significance_level and abs_uplift > 0:
    st.success(f"**Result is Statistically Significant** (p-value: {p_value:.4f})")
    st.markdown(f"We are more than {100 - significance_level*100:.0f}% confident that the variant performs better than the control.")
else:
    st.warning(f"**Result is Not Statistically Significant** (p-value: {p_value:.4f})")
    st.markdown(f"We cannot be confident that the observed difference is real. The revenue projections below are highly uncertain and should be treated with caution.")

# Metrics columns
col1, col2, col3 = st.columns(3)
col1.metric(
    "Relative Uplift",
    f"{rel_uplift_pct:.2f}%",
    delta=f"{abs_uplift*100:.2f} p.p.",
    help="The percentage improvement of the Variant over the Control. The smaller number is the absolute change in percentage points (p.p.)."
)
col2.metric(
    "P-Value",
    f"{p_value:.4f}",
    help="The probability of observing this result by random chance. Lower is better (typically < 0.05 is considered significant)."
)
col3.metric(
    "95% CI (Absolute Uplift)",
    f"[{ci_low*100:.2f}%, {ci_high*100:.2f}%]",
    help="We are 95% confident that the true absolute uplift lies within this range."
)

st.divider()

# --- Revenue Projection Section ---
st.header(f"💰 Revenue Projection Over {forecast_period} Days")
st.markdown(f"Based on an average of **{daily_traffic:,} visitors per day** and a **{decay_rate_pct}% uplift decay**.")

# --- Visualization & Projection Calculation ---
# Create a date range for the forecast period
dates = pd.date_range(start=pd.Timestamp.today(), periods=forecast_period)

# Calculate daily revenue for control
daily_baseline_revenue = daily_traffic * baseline_rate * avg_revenue

# Calculate daily revenue for variant, applying decay
daily_variant_revenues = []
initial_uplift_revenue = daily_traffic * abs_uplift * avg_revenue

for day in range(forecast_period):
    # Linear decay: the total decay is spread out over the period
    daily_decay_factor = (day / (forecast_period -1)) if forecast_period > 1 else 0
    decayed_uplift_revenue = initial_uplift_revenue * (1 - decay_rate * daily_decay_factor)
    daily_variant_revenue = daily_baseline_revenue + decayed_uplift_revenue
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

st.info(f"The Variant curve flattens slightly over time due to the **{decay_rate_pct}%** decay assumption. This provides a more conservative and potentially more realistic forecast.")
