import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(layout="wide", page_title="Supply Chain Forecast Analysis Dashboard")

# Load Data
@st.cache_data
def load_data():
    try:
        # Load data from CSV file
        df = pd.read_csv("enhanced_sample_data.csv", parse_dates=['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty DataFrame with correct columns as fallback
        return pd.DataFrame(columns=['date', 'category', 'product', 'distribution_center', 'supplier', 
                                    'manual_forecast', 'actual_demand', 'stock_level', 'lead_time_days',
                                    'delivery_performance', 'customer_satisfaction', 'allocation_efficiency'])

# Load the data
df = load_data()

# Main Dashboard
st.title("📦 Supply Chain Forecast Analysis Dashboard")
st.subheader("Impact of Manual Forecasting on Resource Allocation and Customer Satisfaction")

# Sidebar Filters
st.sidebar.header("Filter Data")
categories = st.sidebar.multiselect("Select Category", df['category'].unique(), default=df['category'].unique()[0] if not df.empty else None)

# Only proceed with filtering if categories were selected and data exists
if categories and not df.empty:
    date_min = df['date'].min() if not df.empty else None
    date_max = df['date'].max() if not df.empty else None
    
    if date_min and date_max:
        date_range = st.sidebar.date_input("Select Date Range", [date_min, date_max])
    
    products = st.sidebar.multiselect("Select Products", 
                                    df[df['category'].isin(categories)]['product'].unique(), 
                                    default=df[df['category'].isin(categories)]['product'].unique()[0] if not df.empty else None)
    
    distribution_centers = st.sidebar.multiselect("Select Distribution Centers", 
                                               df['distribution_center'].unique(),
                                               default=df['distribution_center'].unique()[0] if not df.empty else None)
    
    # Filtered Data
    filtered_df = df[
        (df['category'].isin(categories)) &
        (df['product'].isin(products)) &
        (df['distribution_center'].isin(distribution_centers)) &
        (df['date'] >= pd.to_datetime(date_range[0])) &
        (df['date'] <= pd.to_datetime(date_range[1]))
    ]
    
    # Calculate forecast accuracy and error metrics
    if not filtered_df.empty:
        filtered_df['forecast_error'] = filtered_df['manual_forecast'] - filtered_df['actual_demand']
        filtered_df['forecast_error_pct'] = abs(filtered_df['forecast_error']) / filtered_df['actual_demand'] * 100
        filtered_df['forecast_accuracy'] = 100 - filtered_df['forecast_error_pct']
        
        # Classify forecast bias
        filtered_df['forecast_bias'] = pd.cut(
            filtered_df['forecast_error'], 
            bins=[-float('inf'), -10, 10, float('inf')],
            labels=['Under-forecast', 'Accurate', 'Over-forecast']
        )
        
        # Calculate stockout (when stock < demand)
        filtered_df['stockout'] = filtered_df['stock_level'] < filtered_df['actual_demand']
        filtered_df['stockout_qty'] = np.maximum(0, filtered_df['actual_demand'] - filtered_df['stock_level'])
        
        # Calculate overstock (when stock > 120% of demand)
        filtered_df['overstock'] = filtered_df['stock_level'] > (filtered_df['actual_demand'] * 1.2)
        filtered_df['overstock_qty'] = np.maximum(0, filtered_df['stock_level'] - filtered_df['actual_demand'])
        
        # Overview Metrics
        st.markdown("### 📊 Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate key metrics
        avg_forecast_accuracy = round(filtered_df['forecast_accuracy'].mean(), 2)
        avg_customer_satisfaction = round(filtered_df['customer_satisfaction'].mean(), 2)
        avg_allocation_efficiency = round(filtered_df['allocation_efficiency'].mean(), 2)
        stockout_rate = round(filtered_df['stockout'].mean() * 100, 2)
        
        # Display metrics
        col1.metric("Average Forecast Accuracy", f"{avg_forecast_accuracy}%")
        col2.metric("Customer Satisfaction (1-5)", avg_customer_satisfaction)
        col3.metric("Resource Allocation Efficiency", f"{avg_allocation_efficiency}%")
        col4.metric("Stockout Rate", f"{stockout_rate}%")
        
        # Forecast Analysis Section
        st.markdown("### 🔄 Forecast vs Actual Demand Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Forecast vs Actual Demand Chart
            fig_line = px.line(
                filtered_df, x="date", y=["manual_forecast", "actual_demand"],
                title="Manual Forecasts vs Actual Demand",
                labels={"value": "Demand", "variable": "Type"},
                color_discrete_map={
                    "manual_forecast": "#FF9999",
                    "actual_demand": "#66BB66"
                }
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            # Forecast Error Analysis
            forecast_errors = filtered_df.groupby('product')['forecast_error_pct'].mean().reset_index()
            forecast_errors = forecast_errors.sort_values('forecast_error_pct', ascending=False)
            
            fig_error = px.bar(
                forecast_errors, x="product", y="forecast_error_pct",
                title="Average Forecast Error by Product (%)",
                labels={"forecast_error_pct": "Error %", "product": "Product"},
                color="forecast_error_pct",
                color_continuous_scale=px.colors.sequential.Reds
            )
            fig_error.update_layout(yaxis_title="Forecast Error (%)")
            st.plotly_chart(fig_error, use_container_width=True)
        
        # Forecast Bias Analysis
        st.markdown("### 🎯 Forecast Bias Analysis")
        st.markdown("Understanding patterns in manual forecasting errors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of forecast bias
            bias_count = filtered_df['forecast_bias'].value_counts().reset_index()
            bias_count.columns = ['Bias Type', 'Count']
            
            fig_bias = px.pie(
                bias_count, values='Count', names='Bias Type',
                title='Distribution of Forecast Bias',
                color='Bias Type',
                color_discrete_map={
                    'Under-forecast': '#3498db',
                    'Accurate': '#2ecc71',
                    'Over-forecast': '#e74c3c'
                }
            )
            st.plotly_chart(fig_bias, use_container_width=True)
        
        with col2:
            # Bias by product
            bias_by_product = filtered_df.groupby(['product', 'forecast_bias']).size().reset_index(name='count')
            
            fig_bias_product = px.bar(
                bias_by_product, x="product", y="count", color="forecast_bias",
                title="Forecast Bias by Product",
                color_discrete_map={
                    'Under-forecast': '#3498db',
                    'Accurate': '#2ecc71',
                    'Over-forecast': '#e74c3c'
                }
            )
            st.plotly_chart(fig_bias_product, use_container_width=True)
        
        # Impact on Customer Satisfaction
        st.markdown("### 😊 Impact on Customer Satisfaction")
        st.markdown("How forecast accuracy affects customer experience")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation between forecast accuracy and customer satisfaction
            fig_corr = px.scatter(
                filtered_df, x="forecast_accuracy", y="customer_satisfaction",
                title="Forecast Accuracy vs Customer Satisfaction",
                color="stockout",
                labels={
                    "forecast_accuracy": "Forecast Accuracy (%)",
                    "customer_satisfaction": "Customer Satisfaction (1-5)",
                    "stockout": "Stockout Occurred"
                },
                trendline="ols"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # Customer satisfaction by forecast bias
            satisfaction_by_bias = filtered_df.groupby('forecast_bias')['customer_satisfaction'].mean().reset_index()
            
            fig_sat_bias = px.bar(
                satisfaction_by_bias, x="forecast_bias", y="customer_satisfaction",
                title="Customer Satisfaction by Forecast Bias",
                color="forecast_bias",
                labels={
                    "forecast_bias": "Forecast Bias",
                    "customer_satisfaction": "Avg. Customer Satisfaction (1-5)"
                },
                color_discrete_map={
                    'Under-forecast': '#3498db',
                    'Accurate': '#2ecc71',
                    'Over-forecast': '#e74c3c'
                }
            )
            st.plotly_chart(fig_sat_bias, use_container_width=True)
        
        # Resource Allocation Impact
        st.markdown("### 📈 Resource Allocation Impact")
        st.markdown("How forecast errors affect inventory management and resource efficiency")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Allocation efficiency by forecast accuracy
            filtered_df['forecast_accuracy_bin'] = pd.cut(
                filtered_df['forecast_accuracy'], 
                bins=[0, 60, 80, 90, 100], 
                labels=['Poor (<60%)', 'Fair (60-80%)', 'Good (80-90%)', 'Excellent (>90%)']
            )
            
            allocation_by_accuracy = filtered_df.groupby('forecast_accuracy_bin')['allocation_efficiency'].mean().reset_index()
            
            fig_allocation = px.bar(
                allocation_by_accuracy, x="forecast_accuracy_bin", y="allocation_efficiency",
                title="Resource Allocation Efficiency by Forecast Accuracy",
                color="allocation_efficiency",
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={
                    "forecast_accuracy_bin": "Forecast Accuracy Range",
                    "allocation_efficiency": "Allocation Efficiency (%)"
                }
            )
            st.plotly_chart(fig_allocation, use_container_width=True)
        
        with col2:
            # Stockout rate by forecast bias
            stockout_by_bias = filtered_df.groupby('forecast_bias')['stockout'].mean().reset_index()
            stockout_by_bias['stockout_rate'] = stockout_by_bias['stockout'] * 100
            
            fig_stockout_bias = px.bar(
                stockout_by_bias, x="forecast_bias", y="stockout_rate",
                title="Stockout Rate by Forecast Bias",
                color="forecast_bias",
                labels={
                    "forecast_bias": "Forecast Bias",
                    "stockout_rate": "Stockout Rate (%)"
                },
                color_discrete_map={
                    'Under-forecast': '#3498db',
                    'Accurate': '#2ecc71',
                    'Over-forecast': '#e74c3c'
                }
            )
            st.plotly_chart(fig_stockout_bias, use_container_width=True)
        
        # Inventory Analysis
        st.markdown("### 📦 Inventory Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stockout Analysis
            st.subheader("⚠️ Stockout Analysis")
            stockout_df = filtered_df[filtered_df['stockout']]
            st.markdown(f"**{len(stockout_df)} stockout incidents** detected")
            
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
                st.success("No stockouts detected in the selected period")
        
        with col2:
            # Overstock Analysis
            st.subheader("💰 Overstock Analysis")
            overstock_df = filtered_df[filtered_df['overstock']]
            st.markdown(f"**{len(overstock_df)} overstock incidents** detected (>20% excess inventory)")
            
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
                st.success("No significant overstock detected in the selected period")
        
        # Time Series Analysis of Forecast Accuracy
        st.markdown("### 📅 Forecast Accuracy Trends Over Time")
        
        # Aggregate forecast accuracy by date
        accuracy_trend = filtered_df.groupby('date')[['forecast_accuracy', 'customer_satisfaction', 'allocation_efficiency']].mean().reset_index()
        
        fig_trend = px.line(
            accuracy_trend, x="date", y=["forecast_accuracy", "customer_satisfaction", "allocation_efficiency"],
            title="Trends in Forecast Accuracy, Customer Satisfaction, and Resource Allocation",
            labels={
                "value": "Score", 
                "variable": "Metric",
                "date": "Date"
            },
            color_discrete_map={
                "forecast_accuracy": "#FF9999",
                "customer_satisfaction": "#66BB66",
                "allocation_efficiency": "#99CCFF"
            }
        )
        # Add a second y-axis for customer satisfaction
        fig_trend.update_layout(
            yaxis=dict(title="Forecast Accuracy / Allocation Efficiency (%)"),
            yaxis2=dict(
                title="Customer Satisfaction (1-5)",
                overlaying="y",
                side="right",
                range=[1, 5]
            )
        )
        # Update traces for the second y-axis
        fig_trend.update_traces(
            yaxis="y2",
            selector=dict(name="customer_satisfaction")
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Recommendations Section
        st.markdown("---")
        st.markdown("### 💡 Key Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Key Findings:
            1. **Forecast Accuracy Impact**: Manual forecasting accuracy directly correlates with customer satisfaction and resource allocation efficiency.
            2. **Bias Patterns**: There's a tendency toward over-forecasting, leading to excess inventory and associated costs.
            3. **Customer Satisfaction**: Stockouts resulting from under-forecasting have the most negative impact on customer satisfaction.
            4. **Resource Efficiency**: Forecast errors (both over and under) lead to suboptimal resource allocation.
            """)
        
        with col2:
            st.markdown("""
            #### Recommendations:
            1. **Review Forecasting Process**: Identify and address systematic bias in manual forecasting.
            2. **Improve Data Visibility**: Ensure forecasters have access to accurate historical data and trends.
            3. **Implement Checks**: Add validation steps to catch extreme forecast deviations.
            4. **Training Program**: Enhance forecasting skills and awareness of impact on customer satisfaction.
            5. **Regular Reviews**: Conduct periodic reviews of forecast accuracy by product and location.
            """)
        
        # Cost Impact Analysis
        st.markdown("### 💰 Cost Impact of Forecasting Errors")
        
        # Calculate estimated costs
        avg_holding_cost_rate = 0.25  # 25% annual holding cost as percentage of inventory value
        avg_stockout_cost_per_unit = 100  # Average cost of lost sales and customer dissatisfaction per unit
        
        # Calculate for filtered data
        filtered_df['holding_cost'] = filtered_df['overstock_qty'] * avg_holding_cost_rate / 12  # Monthly holding cost
        filtered_df['stockout_cost'] = filtered_df['stockout_qty'] * avg_stockout_cost_per_unit
        filtered_df['total_error_cost'] = filtered_df['holding_cost'] + filtered_df['stockout_cost']
        
        # Aggregate by product
        cost_by_product = filtered_df.groupby('product').agg({
            'holding_cost': 'sum',
            'stockout_cost': 'sum',
            'total_error_cost': 'sum'
        }).reset_index()
        
        fig_cost = px.bar(
            cost_by_product, x="product", y=["holding_cost", "stockout_cost"],
            title="Cost Impact of Forecast Errors by Product",
            labels={
                "value": "Cost ($)",
                "variable": "Cost Type",
                "product": "Product"
            },
            color_discrete_map={
                "holding_cost": "#3498db",
                "stockout_cost": "#e74c3c"
            }
        )
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # Total cost impact
        total_holding_cost = filtered_df['holding_cost'].sum()
        total_stockout_cost = filtered_df['stockout_cost'].sum()
        total_error_cost = filtered_df['total_error_cost'].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Holding Cost", f"${total_holding_cost:,.2f}")
        col2.metric("Total Stockout Cost", f"${total_stockout_cost:,.2f}")
        col3.metric("Total Error Cost", f"${total_error_cost:,.2f}")
        
        # Download report section
        st.markdown("---")
        st.markdown("### 📩 Download Analysis Report")
        
        # Create a CSV with the analysis results
        @st.cache_data
        def get_analysis_csv():
            # Summary data
            summary_data = {
                'Metric': [
                    'Average Forecast Accuracy (%)',
                    'Average Customer Satisfaction (1-5)',
                    'Average Resource Allocation Efficiency (%)',
                    'Stockout Rate (%)',
                    'Number of Stockout Incidents',
                    'Number of Overstock Incidents',
                    'Total Holding Cost ($)',
                    'Total Stockout Cost ($)',
                    'Total Error Cost ($)'
                ],
                'Value': [
                    avg_forecast_accuracy,
                    avg_customer_satisfaction,
                    avg_allocation_efficiency,
                    stockout_rate,
                    len(stockout_df),
                    len(overstock_df),
                    total_holding_cost,
                    total_stockout_cost,
                    total_error_cost
                ]
            }
            
            return pd.DataFrame(summary_data)
        
        # Generate report download link
        summary_csv = get_analysis_csv()
        csv_data = summary_csv.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Analysis Summary (CSV)",
            data=csv_data,
            file_name="forecast_analysis_summary.csv",
            mime="text/csv"
        )
        
        # Footer
        st.markdown("---")
        st.caption("Supply Chain Analytics Dashboard | Made with Streamlit")
        st.caption("Last updated: April 2023")
    
    else:
        st.warning("No data available with the selected filters. Please adjust your selection.")
else:
    st.info("Please select at least one category to analyze the data.")