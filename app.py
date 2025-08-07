import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="A/B Test Revenue & ABV Projection",
    page_icon="📈",
    layout="wide"
)

# --- Initialize Session State ---
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
c1, c2 = st.sidebar.columns(2)
with c1:
    baseline_conv = c1.number_input("Control CVR (%)", 0.0, 100.0, 2.5, 0.1, help="The conversion rate of the original version (Control).")
    baseline_users = c1.number_input("Control Visitors", 1, None, 10000, 100, help="Number of unique visitors who saw the Control version.")
with c2:
    variant_conv = c2.number_input("Variant CVR (%)", 0.0, 100.0, 2.8, 0.1, help="The conversion rate of the new version (Variant).")
    variant_users = c2.number_input("Variant Visitors", 1, None, 10000, 100, help="Number of unique visitors who saw the Variant version.")

st.sidebar.header("💰 2. Revenue & Forecast Settings")
c3, c4 = st.sidebar.columns(2)
with c3:
    baseline_avg_revenue = c3.number_input("Control ABV ($)", 0.0, None, 120.0, 5.0, help="The average revenue per conversion for the Control group.")
    baseline_std_dev = c3.number_input("Control ABV Std Dev ($)", 0.0, None, 20.0, 1.0, help="Standard deviation of the booking value for the Control group.")
with c4:
    variant_avg_revenue = c4.number_input("Variant ABV ($)", 0.0, None, 125.0, 5.0, help="The average revenue per conversion for the Variant group.")
    variant_std_dev = c4.number_input("Variant ABV Std Dev ($)", 0.0, None, 22.0, 1.0, help="Standard deviation of the booking value for the Variant group.")

daily_traffic = st.sidebar.number_input("Projected Daily Visitors", 1, None, 5000, 100, help="Expected unique visitors per day going forward.")
forecast_period = st.sidebar.number_input("Forecast Period (days)", 1, None, 90, 1, help="How many days into the future to project revenue.")

use_decay = st.sidebar.checkbox("Account for Uplift Decay?")
decay_rate_pct = 0
if use_decay:
    decay_rate_pct = st.sidebar.slider("Uplift Decay Rate (%)", 0, 100, 10, 5, help="The percentage by which the initial total uplift is expected to decrease by the end of the forecast period.")

st.sidebar.divider()

# --- Buttons ---
col_btn1, col_btn2 = st.sidebar.columns(2)
run_button = col_btn1.button("🚀 Run Calculation", type="primary", use_container_width=True)
reset_button = col_btn2.button("🔄 Reset Inputs", use_container_width=True)

if reset_button:
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

if run_button:
    st.session_state.results_calculated = True
    
    # --- Calculations ---
    baseline_rate = baseline_conv / 100
    variant_rate = variant_conv / 100
    decay_rate = decay_rate_pct / 100
    baseline_conversions_test = baseline_rate * baseline_users
    variant_conversions_test = variant_rate * variant_users

    # CVR Uplift & Significance
    cvr_abs_uplift = variant_rate - baseline_rate
    cvr_rel_uplift_pct = (cvr_abs_uplift / baseline_rate) * 100 if baseline_rate > 0 else np.nan
    p_pool = (baseline_conversions_test + variant_conversions_test) / (baseline_users + variant_users)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / baseline_users + 1 / variant_users)) if p_pool > 0 else 0
    z_score = cvr_abs_uplift / se_pool if se_pool > 0 else 0
    p_value_cvr = stats.norm.sf(abs(z_score)) * 2

    # ABV Uplift & Significance (t-test)
    abv_abs_uplift = variant_avg_revenue - baseline_avg_revenue
    abv_rel_uplift_pct = (abv_abs_uplift / baseline_avg_revenue) * 100 if baseline_avg_revenue > 0 else np.nan
    _, p_value_abv = stats.ttest_ind_from_stats(
        mean1=baseline_avg_revenue, std1=baseline_std_dev, nobs1=baseline_conversions_test,
        mean2=variant_avg_revenue, std2=variant_std_dev, nobs2=variant_conversions_test,
        equal_var=False  # Welch's t-test
    ) if baseline_conversions_test > 1 and variant_conversions_test > 1 else (0, 1.0)

    # Confidence Interval for CVR uplift
    se_diff = np.sqrt(baseline_rate * (1 - baseline_rate) / baseline_users + variant_rate * (1 - variant_rate) / variant_users)
    ci_low_abs = cvr_abs_uplift - stats.norm.ppf(0.975) * se_diff
    ci_high_abs = cvr_abs_uplift + stats.norm.ppf(0.975) * se_diff

    # --- Projection Calculation with Confidence Interval ---
    dates = pd.date_range(start=pd.Timestamp.today(), periods=forecast_period)
    
    def calculate_cumulative_revenue(cvr, abv):
        daily_rev = daily_traffic * cvr * abv
        initial_daily_lift = daily_rev - (daily_traffic * baseline_rate * baseline_avg_revenue)
        
        daily_revenues = []
        for day in range(forecast_period):
            daily_decay_factor = (day / (forecast_period - 1)) if forecast_period > 1 else 0
            decayed_lift = initial_daily_lift * (1 - decay_rate * daily_decay_factor)
            daily_variant_revenue = (daily_traffic * baseline_rate * baseline_avg_revenue) + decayed_lift
            daily_revenues.append(daily_variant_revenue)
        return np.cumsum(daily_revenues)

    control_cumulative = np.cumsum([daily_traffic * baseline_rate * baseline_avg_revenue] * forecast_period)
    variant_cumulative_mean = calculate_cumulative_revenue(variant_rate, variant_avg_revenue)
    
    variant_cumulative_lower = calculate_cumulative_revenue(baseline_rate + ci_low_abs, variant_avg_revenue)
    variant_cumulative_upper = calculate_cumulative_revenue(baseline_rate + ci_high_abs, variant_avg_revenue)

    # Calculate revenue lift confidence interval
    proj_revenue_diff_lower = variant_cumulative_lower[-1] - control_cumulative[-1]
    proj_revenue_diff_upper = variant_cumulative_upper[-1] - control_cumulative[-1]

    projection_df = pd.DataFrame({
        'Date': dates,
        'Control': control_cumulative,
        'Variant': variant_cumulative_mean,
        'Lower Bound': variant_cumulative_lower,
        'Upper Bound': variant_cumulative_upper,
    })

    # --- Store results in session state ---
    st.session_state.update({
        'p_value_cvr': p_value_cvr, 'p_value_abv': p_value_abv,
        'cvr_abs_uplift': cvr_abs_uplift, 'cvr_rel_uplift_pct': cvr_rel_uplift_pct,
        'abv_abs_uplift': abv_abs_uplift, 'abv_rel_uplift_pct': abv_rel_uplift_pct,
        'proj_baseline_revenue': control_cumulative[-1],
        'proj_variant_revenue': variant_cumulative_mean[-1],
        'proj_revenue_diff': variant_cumulative_mean[-1] - control_cumulative[-1],
        'proj_revenue_diff_lower': proj_revenue_diff_lower,
        'proj_revenue_diff_upper': proj_revenue_diff_upper,
        'projection_df': projection_df,
        'decay_rate_pct': decay_rate_pct, 'forecast_period': forecast_period,
        'daily_traffic': daily_traffic
    })

# --- Results Display Area ---
if not st.session_state.results_calculated:
    st.info("Please enter your A/B test data in the sidebar and click 'Run Calculation'.")
else:
    st.header("📊 Key Results Summary")

    def display_significance(p_value, uplift, name):
        st.subheader(f"{name} Significance")
        significance_level = 0.05
        if p_value < significance_level and uplift > 0:
            st.success(f"**Result is Statistically Significant** (p-value: {p_value:.4f})")
            st.markdown(f"We are confident the change in **{name.lower()}** is a real improvement.")
        else:
            st.warning(f"**Result is Not Statistically Significant** (p-value: {p_value:.4f})")
            st.markdown(f"We cannot be confident the change in **{name.lower()}** is real.")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        display_significance(st.session_state.p_value_cvr, st.session_state.cvr_abs_uplift, "Conversion Rate")
    with res_col2:
        display_significance(st.session_state.p_value_abv, st.session_state.abv_abs_uplift, "Average Booking Value")

    st.divider()
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    met_col1.metric("CVR Uplift", f"{st.session_state.cvr_rel_uplift_pct:.2f}%", delta=f"{st.session_state.cvr_abs_uplift*100:.2f} p.p.")
    met_col2.metric("P-Value (CVR)", f"{st.session_state.p_value_cvr:.4f}")
    met_col3.metric("ABV Uplift", f"{st.session_state.abv_rel_uplift_pct:.2f}%", delta=f"${st.session_state.abv_abs_uplift:.2f}")
    met_col4.metric("P-Value (ABV)", f"{st.session_state.p_value_abv:.4f}")
    
    st.divider()
    st.header(f"💰 Revenue Projection Over {st.session_state.forecast_period} Days")
    st.markdown(f"Based on **{st.session_state.daily_traffic:,} daily visitors** and a **{st.session_state.decay_rate_pct}% uplift decay**.")
    
    proj_col1, proj_col2, proj_col3 = st.columns(3)
    proj_col1.metric("Projected Control Revenue", f"${st.session_state.proj_baseline_revenue:,.0f}")
    proj_col2.metric("Projected Variant Revenue", f"${st.session_state.proj_variant_revenue:,.0f}")
    proj_col3.metric("Projected Revenue Lift", f"${st.session_state.proj_revenue_diff:,.0f}")

    # --- Altair Chart with Confidence Band ---
    df_melted = st.session_state.projection_df.melt(id_vars=['Date'], value_vars=['Control', 'Variant', 'Lower Bound', 'Upper Bound'], var_name='Type', value_name='Revenue')
    
    control_line = alt.Chart(df_melted.query("Type == 'Control'")).mark_line(color='gray', strokeDash=[5,5]).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Revenue:Q', title='Cumulative Revenue ($)'),
        tooltip=['Date', 'Revenue']
    )
    
    variant_line = alt.Chart(df_melted.query("Type == 'Variant'")).mark_line(color='#00A699').encode(
        x='Date:T',
        y='Revenue:Q',
        tooltip=['Date', 'Revenue']
    )
    
    confidence_band = alt.Chart(st.session_state.projection_df).mark_area(opacity=0.3, color='#00A699').encode(
        x='Date:T',
        y='Lower Bound:Q',
        y2='Upper Bound:Q'
    )
    
    st.altair_chart(confidence_band + control_line + variant_line, use_container_width=True)

    st.divider()

    # --- Executive Summary Section ---
    st.header("Executive Summary & Recommendation")
    with st.expander("Click to see the final summary"):
        cvr_sig = st.session_state.p_value_cvr < 0.05 and st.session_state.cvr_abs_uplift > 0
        abv_sig = st.session_state.p_value_abv < 0.05 and st.session_state.abv_abs_uplift > 0

        st.markdown(f"""
        Over a **{st.session_state.forecast_period}-day period**, the variant is projected to generate an additional **${st.session_state.proj_revenue_diff:,.0f}** in revenue.
        
        We are 95% confident that the true revenue lift lies between **${st.session_state.proj_revenue_diff_lower:,.0f}** and **${st.session_state.proj_revenue_diff_upper:,.0f}**.
        """)

        if cvr_sig and abv_sig:
            st.success("**Recommendation: Roll out.** The variant showed statistically significant improvements in both conversion rate and average booking value. The financial upside is clear and backed by strong evidence.")
        elif cvr_sig or abv_sig:
            st.warning("**Recommendation: Consider rolling out with caution.** The variant showed a significant improvement in one key metric but not the other. While there is a projected financial gain, the result is not uniformly positive. Evaluate the business case for the specific metric that improved.")
        else:
            st.error("**Recommendation: Do not roll out.** Neither the change in conversion rate nor average booking value was statistically significant. The observed lift is likely due to random chance, and we cannot be confident in a positive financial return.")
