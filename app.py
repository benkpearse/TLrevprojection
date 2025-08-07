import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

st.set_page_config(page_title="A/B Test Revenue Projection", layout="centered")

st.title("📈 A/B Test Revenue Projection Tool")

# Input section
st.header("Input A/B Test Results")
col1, col2 = st.columns(2)
with col1:
    baseline_conv = st.number_input(
        "Baseline Conversion Rate (%)", min_value=0.0, max_value=100.0, value=2.5, step=0.1)
    variant_conv = st.number_input(
        "Variant Conversion Rate (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    baseline_users = st.number_input(
        "Baseline Sample Size", min_value=1, value=10000, step=100)
with col2:
    variant_users = st.number_input(
        "Variant Sample Size", min_value=1, value=10000, step=100)
    avg_revenue = st.number_input(
        "Average Revenue per Conversion", min_value=0.0, value=50.0, step=1.0)

# Forecast parameters
st.header("Forecast Settings")
forecast_period = st.number_input(
    "Forecast Period (days)", min_value=1, value=30, step=1)

# Compute metrics
baseline_rate = baseline_conv / 100
variant_rate = variant_conv / 100

# Uplift calculation
delta = variant_rate - baseline_rate
uplift_pct = (delta / baseline_rate) * 100 if baseline_rate > 0 else np.nan

# Confidence interval for difference in proportions
se = np.sqrt(baseline_rate * (1 - baseline_rate) / baseline_users + 
             variant_rate * (1 - variant_rate) / variant_users)
z = stats.norm.ppf(0.975)
ci_low = delta - z * se
ci_high = delta + z * se
ci_low_pct = ci_low * 100
ci_high_pct = ci_high * 100

# Projected conversions
baseline_conversions = baseline_rate * forecast_period * baseline_users / baseline_users
variant_conversions = variant_rate * forecast_period * variant_users / variant_users

# Projected revenue
baseline_revenue = baseline_conversions * avg_revenue
variant_revenue = variant_conversions * avg_revenue
revenue_diff = variant_revenue - baseline_revenue

# Display results
st.header("Results")
st.metric("Conversion Uplift", f"{uplift_pct:.2f}%", delta=f"{delta*100:.2f}%")
st.write(f"95% CI for uplift: [{ci_low_pct:.2f}%, {ci_high_pct:.2f}%]")

st.metric("Projected Revenue (Baseline)", f"${baseline_revenue:,.2f}")
st.metric("Projected Revenue (Variant)", f"${variant_revenue:,.2f}", delta=f"${revenue_diff:,.2f}")

# Visualization
st.header("Projection Over Time")
dates = pd.date_range(start=pd.Timestamp.today(), periods=forecast_period)

baseline_daily = baseline_rate * avg_revenue * baseline_users
variant_daily = variant_rate * avg_revenue * variant_users
data = pd.DataFrame({
    "Baseline Daily Revenue": np.repeat(baseline_daily, forecast_period),
    "Variant Daily Revenue": np.repeat(variant_daily, forecast_period)
}, index=dates)

st.line_chart(data)

st.info("This tool projects revenue based on your A/B test results. Adjust inputs to explore different scenarios.")
