import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
start_date = '2023-10-01'
end_date = '2024-04-15'
categories = ['Electronics', 'Clothing', 'Food', 'Home Goods', 'Sports', 'Beauty']
products = {
    'Electronics': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Smart Watch', 'Camera'],
    'Clothing': ['T-shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes', 'Sweater'],
    'Food': ['Bread', 'Milk', 'Cheese', 'Eggs', 'Coffee', 'Chocolate'],
    'Home Goods': ['Chair', 'Lamp', 'Table', 'Rug', 'Curtains', 'Pillow'],
    'Sports': ['Running Shoes', 'Fitness Tracker', 'Yoga Mat', 'Dumbbells', 'Tennis Racket'],
    'Beauty': ['Shampoo', 'Perfume', 'Makeup', 'Skincare', 'Hair Dryer']
}

# Define distribution centers
distribution_centers = ['East DC', 'West DC', 'Central DC', 'South DC']
suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E']

# Generate dates
dates = pd.date_range(start=start_date, end=end_date)
data = []

print(f"Generating data from {start_date} to {end_date}...")

# Define seasonality patterns by month for each category
seasonality_by_month = {
    'Electronics': {
        1: 0.8,  # Post-holiday drop
        2: 0.7,
        3: 0.8,
        4: 0.9,
        5: 0.9,
        6: 1.0,
        7: 1.1,  # Back to school beginning
        8: 1.3,  # Back to school peak
        9: 1.1,
        10: 1.2,  # Pre-holiday
        11: 1.7,  # Black Friday/Cyber Monday
        12: 1.8   # Holiday shopping
    },
    'Clothing': {
        1: 0.7,  # Winter sales
        2: 0.8,
        3: 1.1,  # Spring collection
        4: 1.2,
        5: 1.3,  # Summer preparation
        6: 1.4,
        7: 1.2,  # Summer sales
        8: 1.3,  # Back to school
        9: 1.1,  # Fall collection
        10: 1.0,
        11: 1.2,  # Pre-holiday
        12: 1.5   # Holiday shopping
    },
    'Food': {
        1: 0.9,
        2: 0.9,
        3: 1.0,
        4: 1.0,
        5: 1.1,
        6: 1.1,
        7: 1.2,
        8: 1.2,
        9: 1.0,
        10: 1.1,
        11: 1.4,  # Thanksgiving
        12: 1.5   # Holiday parties
    },
    'Home Goods': {
        1: 0.8,  # Post-holiday
        2: 0.9,
        3: 1.1,  # Spring cleaning
        4: 1.2,
        5: 1.3,  # Home improvement season
        6: 1.2,
        7: 1.1,
        8: 1.0,
        9: 0.9,
        10: 1.0,
        11: 1.3,  # Pre-holiday
        12: 1.5   # Holiday decorating
    },
    'Sports': {
        1: 1.3,  # New Year's resolutions
        2: 1.1,
        3: 1.0,
        4: 1.1,  # Spring sports
        5: 1.3,  # Summer preparation
        6: 1.4,
        7: 1.3,
        8: 1.1,
        9: 0.9,
        10: 0.8,
        11: 0.9,  # Holiday gifts
        12: 1.2
    },
    'Beauty': {
        1: 0.8,
        2: 0.9,  # Valentine's Day
        3: 1.0,
        4: 1.0,
        5: 1.1,  # Summer preparation
        6: 1.2,
        7: 1.1,
        8: 1.0,
        9: 1.0,
        10: 1.1,
        11: 1.3,  # Pre-holiday
        12: 1.7   # Holiday gifts
    }
}

# Define base forecast error rates by product category (more realistic forecasting capabilities)
forecast_error_ranges = {
    'Electronics': (0.05, 0.15),  # 5-15% error
    'Clothing': (0.07, 0.18),     # 7-18% error
    'Food': (0.03, 0.10),         # 3-10% error (more predictable)
    'Home Goods': (0.08, 0.20),   # 8-20% error
    'Sports': (0.06, 0.17),       # 6-17% error
    'Beauty': (0.05, 0.15)        # 5-15% error
}

# Define product-specific seasonal effects
product_seasonal_effects = {
    'T-shirt': {'summer_boost': 1.5, 'winter_reduction': 0.7},
    'Dress': {'summer_boost': 1.4, 'winter_reduction': 0.6},
    'Jacket': {'summer_reduction': 0.5, 'winter_boost': 1.6},
    'Sweater': {'summer_reduction': 0.4, 'winter_boost': 1.7},
    'Running Shoes': {'spring_boost': 1.3, 'summer_boost': 1.4},
    'Yoga Mat': {'january_boost': 1.5},  # New Year's resolutions
    'Perfume': {'february_boost': 1.4, 'december_boost': 1.6},  # Valentine's & Christmas
    'Makeup': {'december_boost': 1.5},
    'Laptop': {'august_boost': 1.4, 'november_boost': 1.5},  # Back to school, Black Friday
    'Camera': {'summer_boost': 1.3},  # Travel season
    'Bread': {'steady': True},  # Steady demand
    'Milk': {'steady': True},
    'Chocolate': {'winter_boost': 1.3, 'february_boost': 1.4}  # Valentine's
}

# Market events calendar - specific events that affect demand
market_events = {
    '2023-11-24': {'name': 'Black Friday', 'categories': ['Electronics', 'Clothing', 'Home Goods'], 'boost': 2.5},
    '2023-11-27': {'name': 'Cyber Monday', 'categories': ['Electronics'], 'boost': 2.3},
    '2023-12-15': {'name': 'Holiday Shopping Peak', 'categories': ['all'], 'boost': 1.8},
    '2024-01-01': {'name': 'New Year', 'categories': ['Sports', 'Food'], 'boost': 1.5},
    '2024-02-14': {'name': 'Valentine\'s Day', 'categories': ['Beauty', 'Food'], 'boost': 1.7},
    '2024-03-15': {'name': 'Spring Sales', 'categories': ['Clothing', 'Home Goods'], 'boost': 1.4}
}

# Define supplier reliability characteristics
supplier_characteristics = {
    'Supplier A': {'lead_time_range': (2, 4), 'lead_time_variability': 0.1, 'quality': 'high'},
    'Supplier B': {'lead_time_range': (3, 6), 'lead_time_variability': 0.15, 'quality': 'good'},
    'Supplier C': {'lead_time_range': (4, 8), 'lead_time_variability': 0.2, 'quality': 'average'},
    'Supplier D': {'lead_time_range': (5, 12), 'lead_time_variability': 0.25, 'quality': 'variable'},
    'Supplier E': {'lead_time_range': (8, 18), 'lead_time_variability': 0.3, 'quality': 'low'}
}

# Define distribution center characteristics
dc_characteristics = {
    'East DC': {'forecast_bias': 'slight_under', 'efficiency': 'high'},
    'West DC': {'forecast_bias': 'slight_over', 'efficiency': 'high'},
    'Central DC': {'forecast_bias': 'neutral', 'efficiency': 'medium'},
    'South DC': {'forecast_bias': 'variable', 'efficiency': 'low'}
}

# Forecaster profiles (simulates different people making forecasts)
forecaster_profiles = {
    'conservative': {'bias': -0.05, 'error_multiplier': 0.9},  # Tends to under-forecast by 5%
    'optimistic': {'bias': 0.08, 'error_multiplier': 1.1},     # Tends to over-forecast by 8%
    'accurate': {'bias': 0.0, 'error_multiplier': 0.7},        # No bias, lower errors
    'variable': {'bias': 0.0, 'error_multiplier': 1.3},        # No bias, higher errors
    'seasonal_naive': {'bias': 0.0, 'error_multiplier': 1.2}   # More affected by seasonality
}

# Assign forecaster profiles to distribution centers
dc_forecaster = {
    'East DC': 'conservative',
    'West DC': 'optimistic',
    'Central DC': 'accurate',
    'South DC': 'variable'
}

for date in dates:
    # Check if there's a market event on this date
    event_boost = 1.0
    event_categories = []
    if date.strftime('%Y-%m-%d') in market_events:
        event = market_events[date.strftime('%Y-%m-%d')]
        event_categories = event['categories'] if event['categories'] != ['all'] else categories
        event_boost = event['boost']
    
    for category in categories:
        # Season factor from the month-category matrix
        month_factor = seasonality_by_month[category][date.month]
        
        # Only generate data for a subset of products each day to reduce dataset size
        selected_products = random.sample(products[category], min(3, len(products[category])))
        
        for product in selected_products:
            # Only generate data for some distribution centers each day
            selected_dcs = random.sample(distribution_centers, min(2, len(distribution_centers)))
            
            for dc in selected_dcs:
                supplier = random.choice(suppliers)
                
                # Base demand varies by product and has seasonal patterns
                if category == 'Electronics':
                    base_demand = np.random.randint(80, 200)
                elif category == 'Clothing':
                    base_demand = np.random.randint(100, 300)
                elif category == 'Food':
                    base_demand = np.random.randint(150, 400)
                elif category == 'Sports':
                    base_demand = np.random.randint(60, 180)
                elif category == 'Beauty':
                    base_demand = np.random.randint(70, 220)
                else:  # Home Goods
                    base_demand = np.random.randint(50, 150)
                
                # Apply monthly seasonality
                seasonal_factor = month_factor
                
                # Apply product-specific seasonality
                if product in product_seasonal_effects:
                    effects = product_seasonal_effects[product]
                    
                    if 'steady' in effects and effects['steady']:
                        # Reduce seasonality for steady products
                        seasonal_factor = 1.0 + (seasonal_factor - 1.0) * 0.3
                    
                    if 'summer_boost' in effects and date.month in [6, 7, 8]:
                        seasonal_factor *= effects['summer_boost']
                    
                    if 'winter_boost' in effects and date.month in [12, 1, 2]:
                        seasonal_factor *= effects['winter_boost']
                    
                    if 'summer_reduction' in effects and date.month in [6, 7, 8]:
                        seasonal_factor *= effects['summer_reduction']
                    
                    if 'winter_reduction' in effects and date.month in [12, 1, 2]:
                        seasonal_factor *= effects['winter_reduction']
                    
                    if 'spring_boost' in effects and date.month in [3, 4, 5]:
                        seasonal_factor *= effects['spring_boost']
                    
                    if f"{date.month_name().lower()}_boost" in effects:
                        month_name = date.month_name().lower()
                        if f"{month_name}_boost" in effects:
                            seasonal_factor *= effects[f"{month_name}_boost"]
                
                # Apply weekend effect (less pronounced for B2B products)
                if date.weekday() >= 5:  # Weekend
                    if category in ['Food', 'Beauty', 'Clothing']:  # More consumer-focused
                        seasonal_factor *= 1.2
                    else:
                        seasonal_factor *= 1.05  # Less weekend effect for B2B
                
                # Apply event boost if applicable
                if category in event_categories:
                    seasonal_factor *= event_boost
                
                # Add some randomness to demand (business noise)
                demand_noise = np.random.normal(0, 0.03)  # 3% standard deviation
                
                # Actual demand (ground truth with controlled randomness)
                actual_demand = int(base_demand * seasonal_factor * (1 + demand_noise))
                actual_demand = max(5, actual_demand)  # Ensure minimum demand
                
                # MANUAL FORECAST - more realistic error patterns
                
                # Get forecaster profile based on distribution center
                forecaster_type = dc_forecaster[dc]
                forecaster = forecaster_profiles[forecaster_type]
                
                # Get base error range for this category
                min_error, max_error = forecast_error_ranges[category]
                
                # Apply forecaster's error multiplier
                adjusted_min_error = min_error * forecaster['error_multiplier']
                adjusted_max_error = max_error * forecaster['error_multiplier']
                
                # Generate forecast error with forecaster's bias
                manual_error = np.random.uniform(adjusted_min_error, adjusted_max_error)
                
                # Determine direction of error (with bias consideration)
                error_direction = 1 if np.random.random() > (0.5 - forecaster['bias']) else -1
                manual_error *= error_direction
                
                # For event days, increase error slightly (harder to forecast accurately)
                if category in event_categories:
                    manual_error *= 1.2
                
                # Apply the error to get forecast
                manual_forecast = int(actual_demand * (1 + manual_error))
                manual_forecast = max(3, manual_forecast)  # Ensure minimum forecast value
                
                # Calculate forecast accuracy metrics
                forecast_error = manual_forecast - actual_demand
                forecast_error_pct = abs(forecast_error) / max(1, actual_demand) * 100
                forecast_accuracy = 100 - forecast_error_pct
                
                # Calculate stock level based on manual forecast with variable safety stock
                # More accurate forecasters use lower safety stock
                safety_stock_factor = 0.10  # Default 10% safety stock
                
                if forecast_accuracy >= 92:
                    # Very good forecast - minimal safety stock needed
                    safety_stock_factor = np.random.uniform(0.05, 0.08)
                elif forecast_accuracy >= 85:
                    # Good forecast
                    safety_stock_factor = np.random.uniform(0.08, 0.12)
                elif forecast_accuracy >= 75:
                    # Average forecast
                    safety_stock_factor = np.random.uniform(0.12, 0.18)
                else:
                    # Poor forecast - need more safety stock
                    safety_stock_factor = np.random.uniform(0.18, 0.25)
                
                # Add more safety stock for items with long lead times
                supplier_info = supplier_characteristics[supplier]
                min_lead, max_lead = supplier_info['lead_time_range']
                if min_lead > 5:  # Long lead time supplier
                    safety_stock_factor += 0.05
                
                # Calculate stock level
                stock_level = max(3, int(manual_forecast * (1 + safety_stock_factor)))
                
                # Calculate lead time (days from order to delivery)
                base_lead_time = random.randint(supplier_info['lead_time_range'][0], 
                                               supplier_info['lead_time_range'][1])
                
                # Add variability to lead time
                lead_time_variability = supplier_info['lead_time_variability']
                lead_time = max(1, int(base_lead_time * (1 + np.random.normal(0, lead_time_variability))))
                
                # Calculate delivery performance (on-time = 100, early/late = lower)
                scheduled_delivery_days = base_lead_time
                actual_delivery_days = lead_time
                delivery_variance = actual_delivery_days - scheduled_delivery_days
                
                if delivery_variance == 0:  # On time
                    delivery_performance = 100
                elif delivery_variance < 0:  # Early
                    # Being early is good but not as good as being exactly on time
                    delivery_performance = 95 - delivery_variance  # Bonus for early
                else:  # Late
                    # Severe penalty for being late
                    delivery_performance = max(40, 100 - delivery_variance * 8)
                
                # Calculate stockout situation (lower probability with better forecasting)
                stockout_quantity = max(0, actual_demand - stock_level)
                is_stockout = stockout_quantity > 0
                
                # Stockout severity classification
                stockout_severity = 0
                if is_stockout:
                    stockout_ratio = stockout_quantity / actual_demand
                    if stockout_ratio <= 0.1:
                        stockout_severity = 1  # Minor
                    elif stockout_ratio <= 0.3:
                        stockout_severity = 2  # Moderate
                    else:
                        stockout_severity = 3  # Severe
                
                # IMPROVED CUSTOMER SATISFACTION MODEL - More directly linked to forecast accuracy and stockouts
                
                # Base customer satisfaction (1-5 scale)
                # This creates a strong correlation between forecast accuracy and satisfaction
                if forecast_accuracy >= 95:
                    base_satisfaction = np.random.uniform(4.7, 5.0)
                elif forecast_accuracy >= 90:
                    base_satisfaction = np.random.uniform(4.3, 4.8)
                elif forecast_accuracy >= 85:
                    base_satisfaction = np.random.uniform(4.0, 4.5)
                elif forecast_accuracy >= 80:
                    base_satisfaction = np.random.uniform(3.7, 4.2)
                elif forecast_accuracy >= 75:
                    base_satisfaction = np.random.uniform(3.3, 3.8)
                elif forecast_accuracy >= 70:
                    base_satisfaction = np.random.uniform(3.0, 3.5)
                else:
                    base_satisfaction = np.random.uniform(2.5, 3.2)
                
                # Adjust for stockouts - stockouts should have significant impact on satisfaction
                if is_stockout:
                    if stockout_severity == 1:
                        stockout_adjustment = np.random.uniform(-0.8, -0.5)  # Minor impact
                    elif stockout_severity == 2:
                        stockout_adjustment = np.random.uniform(-1.5, -1.0)  # Moderate impact
                    else:
                        stockout_adjustment = np.random.uniform(-2.5, -1.8)  # Severe impact
                    base_satisfaction += stockout_adjustment
                
                # Adjust for delivery performance
                if delivery_performance >= 95:
                    delivery_adjustment = np.random.uniform(0.1, 0.3)
                elif delivery_performance >= 85:
                    delivery_adjustment = np.random.uniform(0, 0.1)
                elif delivery_performance >= 70:
                    delivery_adjustment = np.random.uniform(-0.3, -0.1)
                else:
                    delivery_adjustment = np.random.uniform(-0.7, -0.3)
                base_satisfaction += delivery_adjustment
                
                # Adjust for quality based on supplier
                quality_adjustment = 0
                if supplier_info['quality'] == 'high':
                    quality_adjustment = np.random.uniform(0.1, 0.3)
                elif supplier_info['quality'] == 'good':
                    quality_adjustment = np.random.uniform(0.05, 0.15)
                elif supplier_info['quality'] == 'low':
                    quality_adjustment = np.random.uniform(-0.3, -0.1)
                base_satisfaction += quality_adjustment
                
                # Ensure satisfaction is within bounds
                customer_satisfaction = max(1.0, min(5.0, base_satisfaction))
                
                # Calculate allocation efficiency - improve the formula
                # The goal is to have efficiency close to 100% when stock_level closely matches demand
                # Perfect efficiency = stock just matches demand
                
                if actual_demand == 0:
                    # Edge case - no demand
                    allocation_efficiency = 0 if stock_level > 0 else 100
                else:
                    # Calculate how perfectly stock matches demand
                    stock_ratio = stock_level / actual_demand
                    
                    if 0.95 <= stock_ratio <= 1.05:
                        # Nearly perfect allocation (within 5%)
                        allocation_efficiency = np.random.uniform(95, 100)
                    elif 0.9 <= stock_ratio <= 1.1:
                        # Very good allocation (within 10%)
                        allocation_efficiency = np.random.uniform(90, 95)
                    elif 0.8 <= stock_ratio <= 1.2:
                        # Good allocation (within 20%)
                        allocation_efficiency = np.random.uniform(80, 90)
                    elif 0.7 <= stock_ratio <= 1.3:
                        # Decent allocation (within 30%)
                        allocation_efficiency = np.random.uniform(70, 80)
                    elif 0.5 <= stock_ratio <= 1.5:
                        # Poor allocation (within 50%)
                        allocation_efficiency = np.random.uniform(50, 70)
                    else:
                        # Very poor allocation (more than 50% off)
                        allocation_efficiency = np.random.uniform(20, 50)
                
                # Generate record with the metrics needed by the dashboard
                data.append({
                    'date': date,
                    'category': category,
                    'product': product,
                    'distribution_center': dc,
                    'supplier': supplier,
                    'manual_forecast': manual_forecast,
                    'actual_demand': actual_demand,
                    'stock_level': stock_level,
                    'lead_time_days': lead_time,
                    'delivery_performance': round(delivery_performance, 2),
                    'customer_satisfaction': round(customer_satisfaction, 2),
                    'allocation_efficiency': round(allocation_efficiency, 2),
                    'forecast_accuracy': round(forecast_accuracy, 2),
                    'stockout': is_stockout,
                    'stockout_qty': stockout_quantity,
                    'overstock': stock_level > (actual_demand * 1.2),
                    'overstock_qty': max(0, stock_level - actual_demand),
                    'forecast_error': forecast_error,
                    'forecast_error_pct': round(forecast_error_pct, 2)
                })

# Create DataFrame
df = pd.DataFrame(data)

# Add forecast bias classification with more realistic thresholds
df['forecast_bias'] = pd.cut(
    df['forecast_error'], 
    bins=[-float('inf'), -15, 15, float('inf')],
    labels=['Under-forecast', 'Accurate', 'Over-forecast']
)

# Calculate summary statistics
avg_forecast_accuracy = df['forecast_accuracy'].mean()
avg_satisfaction = df['customer_satisfaction'].mean()
avg_allocation = df['allocation_efficiency'].mean()
avg_lead_time = df['lead_time_days'].mean()
avg_delivery_perf = df['delivery_performance'].mean()
total_stockouts = df['stockout'].sum()
stockout_percentage = total_stockouts / len(df) * 100

# Print summary statistics
print(f"Dataset Statistics:")
print(f"Date Range: {start_date} to {end_date}")
print(f"Total Records: {len(df)}")
print(f"Average Forecast Accuracy: {avg_forecast_accuracy:.2f}%")
print(f"Average Customer Satisfaction (1-5): {avg_satisfaction:.2f}")
print(f"Average Allocation Efficiency: {avg_allocation:.2f}%")
print(f"Average Lead Time: {avg_lead_time:.2f} days")
print(f"Average Delivery Performance: {avg_delivery_perf:.2f}%")
print(f"Stockout Incidents: {total_stockouts} ({stockout_percentage:.2f}%)")

# Bias analysis
bias_distribution = df['forecast_bias'].value_counts(normalize=True) * 100
print("\nForecast Bias Distribution:")
for bias, percentage in bias_distribution.items():
    print(f"{bias}: {percentage:.2f}%")

# Calculate correlation between key metrics to verify relationships
print("\nCorrelation Analysis:")
correlation_matrix = df[['forecast_accuracy', 'customer_satisfaction', 'allocation_efficiency', 'delivery_performance']].corr()
print(correlation_matrix)

# Calculate satisfaction by forecast bias
satisfaction_by_bias = df.groupby('forecast_bias')['customer_satisfaction'].mean()
print("\nAverage Customer Satisfaction by Forecast Bias:")
print(satisfaction_by_bias)

# Calculate stockout rate by forecast bias
stockout_by_bias = df.groupby('forecast_bias')['stockout'].mean() * 100
print("\nStockout Rate by Forecast Bias (%):")
print(stockout_by_bias)

# Calculate efficiency by forecast accuracy bands
df['forecast_accuracy_band'] = pd.cut(
    df['forecast_accuracy'],
    bins=[0, 70, 80, 90, 100],
    labels=['Poor (<70%)', 'Fair (70-80%)', 'Good (80-90%)', 'Excellent (90-100%)']
)

efficiency_by_accuracy = df.groupby('forecast_accuracy_band')['allocation_efficiency'].mean()
print("\nAllocation Efficiency by Forecast Accuracy Band:")
print(efficiency_by_accuracy)

# Save the dataset
output_file = "enhanced_sample.csv"
df.to_csv(output_file, index=False)
print(f"Dataset saved to {output_file}")