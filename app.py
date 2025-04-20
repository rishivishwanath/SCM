import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np

# Page configuration
st.set_page_config(layout="wide", page_title="Supply Chain Forecast Dashboard")

# Load Data
df = pd.read_csv("supply_chain_forecast.csv", parse_dates=['date'])

# Sidebar Filters
st.sidebar.header("Filter Data")
categories = st.sidebar.multiselect("Select Category", df['category'].unique(), default=df['category'].unique()[0])
date_range = st.sidebar.date_input("Select Date Range", [df['date'].min(), df['date'].max()])
products = st.sidebar.multiselect("Select Products", 
                                 df[df['category'].isin(categories)]['product'].unique(), 
                                 default=df[df['category'].isin(categories)]['product'].unique()[0])

# Filtered Data
filtered_df = df[
    (df['category'].isin(categories)) &
    (df['product'].isin(products)) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
]

# Main Dashboard
st.title("📦 Supply Chain Forecast Dashboard")
st.subheader("Evaluating Current Manual Forecasting vs Potential ML Improvements")

# Overview Metrics
st.markdown("### 📊 Overview Metrics")
col1, col2, col3, col4 = st.columns(4)

# Calculate key metrics
manual_forecast_total = filtered_df['manual_forecast'].sum()
ml_forecast_total = filtered_df['ml_forecast'].sum()
actual_demand_total = filtered_df['actual_demand'].sum()

manual_accuracy = round((1 - abs(manual_forecast_total - actual_demand_total) / actual_demand_total) * 100, 2)
ml_accuracy = round((1 - abs(ml_forecast_total - actual_demand_total) / actual_demand_total) * 100, 2)
avg_customer_satisfaction = round(filtered_df['customer_satisfaction'].mean(), 2)
avg_allocation_efficiency = round(filtered_df['allocation_efficiency'].mean(), 2)

# Display metrics with delta (difference between ML and manual)
col1.metric("Current Manual Forecast Accuracy", f"{manual_accuracy}%")
col2.metric("Potential ML Forecast Accuracy", f"{ml_accuracy}%", f"+{round(ml_accuracy - manual_accuracy, 2)}%")
col3.metric("Current Customer Satisfaction (1-5)", avg_customer_satisfaction)
col4.metric("Current Resource Allocation Efficiency", f"{avg_allocation_efficiency}%")

# Forecast Comparison Section
st.markdown("### 🔄 Forecast Comparison")
st.markdown("Compare current manual forecasts (used for inventory decisions) with potential ML-based forecasts")

col1, col2 = st.columns(2)

with col1:
    # Forecast vs Actual Demand Chart
    fig_line = px.line(
        filtered_df, x="date", y=["manual_forecast", "ml_forecast", "actual_demand"],
        title="Current Manual Forecasts vs Potential ML Forecasts",
        labels={"value": "Demand", "variable": "Type"},
        color_discrete_map={
            "manual_forecast": "#FF9999",
            "ml_forecast": "#99CCFF",
            "actual_demand": "#66BB66"
        }
    )
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    # Forecast Error Analysis
    filtered_df['manual_error_pct'] = abs(filtered_df['manual_forecast'] - filtered_df['actual_demand']) / filtered_df['actual_demand'] * 100
    filtered_df['ml_error_pct'] = abs(filtered_df['ml_forecast'] - filtered_df['actual_demand']) / filtered_df['actual_demand'] * 100
    
    # Group by product for comparison
    error_by_product = filtered_df.groupby('product')[['manual_error_pct', 'ml_error_pct']].mean().reset_index()
    
    fig_error = px.bar(
        error_by_product, x="product", y=["manual_error_pct", "ml_error_pct"],
        title="Current vs Potential Forecast Error by Product (%)",
        labels={"value": "Error %", "variable": "Forecast Method"},
        barmode="group",
        color_discrete_map={
            "manual_error_pct": "#FF9999", 
            "ml_error_pct": "#99CCFF"
        }
    )
    fig_error.update_layout(yaxis_title="Forecast Error (%)")
    st.plotly_chart(fig_error, use_container_width=True)

# Business Impact Analysis
st.markdown("### 💼 Current Business Impact Analysis")
st.markdown("Understand how current manual forecast accuracy affects resource allocation and customer satisfaction")

col1, col2 = st.columns(2)

with col1:
    # Customer Satisfaction by Forecast Accuracy
    filtered_df['forecast_accuracy_bin'] = pd.cut(
        filtered_df['manual_error_pct'], 
        bins=[0, 5, 10, 20, 50, 100], 
        labels=['Excellent (<5%)', 'Good (5-10%)', 'Average (10-20%)', 'Poor (20-50%)', 'Very Poor (>50%)']
    )
    
    satisfaction_by_accuracy = filtered_df.groupby('forecast_accuracy_bin')['customer_satisfaction'].mean().reset_index()
    
    fig_satisfaction = px.bar(
        satisfaction_by_accuracy, x="forecast_accuracy_bin", y="customer_satisfaction",
        title="Customer Satisfaction by Current Manual Forecast Accuracy",
        color="customer_satisfaction",
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={"customer_satisfaction": "Satisfaction (1-5)"}
    )
    fig_satisfaction.update_layout(xaxis_title="Forecast Error Range", yaxis_title="Avg. Customer Satisfaction")
    st.plotly_chart(fig_satisfaction, use_container_width=True)

with col2:
    # Resource Allocation Efficiency
    allocation_by_accuracy = filtered_df.groupby('forecast_accuracy_bin')['allocation_efficiency'].mean().reset_index()
    
    fig_allocation = px.bar(
        allocation_by_accuracy, x="forecast_accuracy_bin", y="allocation_efficiency",
        title="Resource Allocation Efficiency by Current Manual Forecast Accuracy",
        color="allocation_efficiency",
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={"allocation_efficiency": "Efficiency (%)"}
    )
    fig_allocation.update_layout(xaxis_title="Forecast Error Range", yaxis_title="Resource Allocation Efficiency (%)")
    st.plotly_chart(fig_allocation, use_container_width=True)

# Inventory Analysis
st.markdown("### 📦 Current Inventory Analysis (Based on Manual Forecasts)")

col1, col2 = st.columns(2)

with col1:
    # Stockout Risk Table
    st.subheader("⚠️ Current Stockout Analysis")
    stockout_df = filtered_df[filtered_df['stock_level'] < filtered_df['actual_demand']]
    st.markdown(f"**{len(stockout_df)} stockout incidents** detected using current manual forecasting")
    
    if not stockout_df.empty:
        stockout_impact = stockout_df.groupby('product').agg({
            'date': 'count',
            'customer_satisfaction': 'mean',
            'actual_demand': 'sum',
            'stock_level': 'sum'
        }).reset_index()
        
        stockout_impact.columns = ['Product', 'Stockout Incidents', 'Avg Satisfaction', 'Total Demand', 'Total Stock']
        stockout_impact['Unfulfilled Demand'] = stockout_impact['Total Demand'] - stockout_impact['Total Stock']
        stockout_impact['Unfulfilled %'] = round(stockout_impact['Unfulfilled Demand'] / stockout_impact['Total Demand'] * 100, 1)
        
        st.dataframe(stockout_impact)
    else:
        st.success("No stockouts detected with current manual forecasting in the selected period")

with col2:
    # Overstock Analysis
    st.subheader("💰 Current Overstock Analysis")
    overstock_df = filtered_df[filtered_df['stock_level'] > filtered_df['actual_demand'] * 1.2]  # 20% over actual demand
    st.markdown(f"**{len(overstock_df)} overstock incidents** detected with current manual forecasting (>20% excess inventory)")
    
    if not overstock_df.empty:
        overstock_impact = overstock_df.groupby('product').agg({
            'date': 'count',
            'actual_demand': 'sum',
            'stock_level': 'sum'
        }).reset_index()
        
        overstock_impact.columns = ['Product', 'Overstock Incidents', 'Total Demand', 'Total Stock']
        overstock_impact['Excess Inventory'] = overstock_impact['Total Stock'] - overstock_impact['Total Demand']
        overstock_impact['Excess %'] = round(overstock_impact['Excess Inventory'] / overstock_impact['Total Demand'] * 100, 1)
        
        st.dataframe(overstock_impact)
    else:
        st.success("No significant overstock detected with current manual forecasting in the selected period")

# ML-Based Forecast Section
st.markdown("### 🤖 Potential ML-Based Demand Forecasting")
st.markdown("What we could achieve by implementing ML forecasting to replace manual forecasting")

# Aggregate demand by date
daily_df = filtered_df.groupby("date")[["actual_demand"]].sum().reset_index()
daily_df.columns = ["ds", "y"]  # Prophet needs these column names

# Initialize & train the model
with st.spinner("Training ML model..."):
    model = Prophet(interval_width=0.95)
    model.fit(daily_df)

    # Create future dataframe (next 14 days)
    future_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=30, value=14)
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)

col1, col2 = st.columns([2, 1])

with col1:
    # Plot forecast
    fig_forecast = plot_plotly(model, forecast)
    fig_forecast.update_layout(title="Potential ML-Based Demand Forecast")
    st.plotly_chart(fig_forecast, use_container_width=True)

with col2:
    # Show predicted values in table
    st.subheader("📋 ML-Forecasted Demand (Next 14 Days)")
    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_days)
    forecast_display.columns = ['Date', 'Predicted Demand', 'Lower Bound', 'Upper Bound']
    forecast_display = forecast_display.round(0)
    st.dataframe(forecast_display)
    
    # Calculate potential improvements
    potential_improvement = ml_accuracy - manual_accuracy
    st.metric("Potential Accuracy Improvement", f"+{round(potential_improvement, 2)}%")
    
    # Recommendations based on forecast
    st.subheader("💡 Recommendations")
    st.markdown("""
    If ML forecasting were implemented:
    1. **Reduce stockouts** by improving prediction accuracy
    2. **Minimize overstock** and carrying costs
    3. **Optimize resource allocation** for peak periods
    4. **Improve customer satisfaction** through better product availability
    """)

# Impact Analysis - What-if Simulation
st.markdown("### 📈 What-if Simulation: Implementing ML Forecasting")
st.markdown("Simulating business outcomes if we switch from manual to ML forecasting")

# Simulation of customer satisfaction with different forecast methods
sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    if 'customer_satisfaction' in filtered_df.columns:
        # Group by date to get average
        satisfaction_by_date = filtered_df.groupby('date')['customer_satisfaction'].mean().reset_index()
        
        # Simulate improved satisfaction with ML forecast
        improved_satisfaction = satisfaction_by_date.copy()
        # Assume ML forecast improves satisfaction proportionally to accuracy improvement
        accuracy_improvement_factor = ml_accuracy / max(1, manual_accuracy)
        improved_satisfaction['improved_satisfaction'] = satisfaction_by_date['customer_satisfaction'].apply(
            lambda x: min(5, x * (1 + (accuracy_improvement_factor - 1) * 0.5))
        )
        
        fig_impact = px.line(
            improved_satisfaction, x="date", 
            y=["customer_satisfaction", "improved_satisfaction"],
            title="Customer Satisfaction: Current (Manual) vs. Potential (ML)",
            labels={
                "value": "Customer Satisfaction (1-5)",
                "variable": "Scenario"
            },
            color_discrete_map={
                "customer_satisfaction": "#FF9999",
                "improved_satisfaction": "#99CCFF"
            }
        )
        fig_impact.update_layout(yaxis_range=[1, 5])
        st.plotly_chart(fig_impact, use_container_width=True)

with sim_col2:
    # Calculate potential revenue impact
    avg_order_value = 100  # Assumed average order value
    current_satisfaction = filtered_df['customer_satisfaction'].mean()
    satisfaction_improvement = min(5, current_satisfaction * accuracy_improvement_factor) - current_satisfaction
    
    # Simple model: 1 point in satisfaction = 5% revenue increase
    revenue_impact_pct = satisfaction_improvement * 5
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = avg_allocation_efficiency + (ml_accuracy - manual_accuracy),
        title = {'text': "Projected Resource Allocation Efficiency with ML"},
        delta = {'reference': avg_allocation_efficiency, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#99CCFF"},
            'steps': [
                {'range': [0, 50], 'color': "#FF9999"},
                {'range': [50, 75], 'color': "#FFCC99"},
                {'range': [75, 100], 'color': "#99CC99"}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': avg_allocation_efficiency
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    ### Estimated Business Impact from ML Implementation:
    - **Customer Satisfaction Improvement**: +{round(satisfaction_improvement, 2)} points
    - **Potential Revenue Impact**: +{round(revenue_impact_pct, 1)}%
    - **Resource Allocation Efficiency Gain**: +{round(ml_accuracy - manual_accuracy, 1)}%
    """)

# Inventory Impact Simulation
st.markdown("### 📊 Simulated Inventory Impact with ML Forecasting")

# Calculate potential stockout reduction
num_current_stockouts = len(filtered_df[filtered_df['stock_level'] < filtered_df['actual_demand']])

# Calculate theoretical stockouts if ML forecasting was used
# Assuming ML forecast accuracy translates to proportional improvement in stock levels
ml_stock_level = filtered_df['manual_forecast'] * (ml_accuracy / manual_accuracy)
num_ml_stockouts = len(filtered_df[ml_stock_level < filtered_df['actual_demand']])

# Calculate potential overstock reduction
num_current_overstock = len(filtered_df[filtered_df['stock_level'] > filtered_df['actual_demand'] * 1.2])
num_ml_overstock = len(filtered_df[ml_stock_level > filtered_df['actual_demand'] * 1.2])

col1, col2 = st.columns(2)

with col1:
    # Stockout comparison
    stockout_data = pd.DataFrame({
        'Method': ['Current Manual Forecasting', 'Potential ML Forecasting'],
        'Stockout Incidents': [num_current_stockouts, num_ml_stockouts]
    })
    
    fig_stockout = px.bar(
        stockout_data, x="Method", y="Stockout Incidents",
        title="Stockout Incidents: Current vs Potential",
        color="Method",
        color_discrete_map={
            "Current Manual Forecasting": "#FF9999",
            "Potential ML Forecasting": "#99CCFF"
        }
    )
    st.plotly_chart(fig_stockout, use_container_width=True)

with col2:
    # Overstock comparison
    overstock_data = pd.DataFrame({
        'Method': ['Current Manual Forecasting', 'Potential ML Forecasting'],
        'Overstock Incidents': [num_current_overstock, num_ml_overstock]
    })
    
    fig_overstock = px.bar(
        overstock_data, x="Method", y="Overstock Incidents",
        title="Overstock Incidents: Current vs Potential",
        color="Method",
        color_discrete_map={
            "Current Manual Forecasting": "#FF9999",
            "Potential ML Forecasting": "#99CCFF"
        }
    )
    st.plotly_chart(fig_overstock, use_container_width=True)

# Conclusion and Recommendations
st.markdown("---")
st.markdown("### 🎯 Key Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Current State Insights:
    1. **Manual forecasting** is currently used for inventory decisions
    2. Current forecast accuracy is **{manual_accuracy}%**
    3. This results in **{stockouts}** stockouts and **{overstocks}** overstock incidents
    4. Customer satisfaction averages **{satisfaction}/5**
    """.format(
        manual_accuracy=manual_accuracy,
        stockouts=num_current_stockouts,
        overstocks=num_current_overstock,
        satisfaction=avg_customer_satisfaction
    ))

with col2:
    st.markdown("""
    #### Potential ML Implementation Benefits:
    1. **Forecast accuracy improvement** of +{accuracy_gain}%
    2. **Stockout reduction** of {stockout_reduction}%
    3. **Overstock reduction** of {overstock_reduction}%
    4. **Customer satisfaction improvement** of +{satisfaction_gain} points
    5. **Revenue potential increase** of +{revenue_gain}%
    """.format(
        accuracy_gain=round(ml_accuracy - manual_accuracy, 2),
        stockout_reduction=round((num_current_stockouts - num_ml_stockouts) / max(1, num_current_stockouts) * 100, 1),
        overstock_reduction=round((num_current_overstock - num_ml_overstock) / max(1, num_current_overstock) * 100, 1),
        satisfaction_gain=round(satisfaction_improvement, 2),
        revenue_gain=round(revenue_impact_pct, 1)
    ))

st.markdown("---")
st.markdown("### 💡 Implementation Recommendations")
st.markdown("""
1. **Pilot ML forecasting** in highest error product categories first
2. **Phase implementation** with A/B testing to validate improvements
3. **Maintain human oversight** of ML forecasts initially
4. **Gradually transition** from manual to ML-based forecasting
5. **Train staff** on interpreting and working with ML forecast data
""")

st.markdown("---")
st.caption("Supply Chain Management Dashboard | Made with Streamlit")