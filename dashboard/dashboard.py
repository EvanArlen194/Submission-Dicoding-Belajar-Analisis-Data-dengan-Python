import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import FuncFormatter

# Set Streamlit Page Config
st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    day_df = pd.read_csv("dashboard/day.csv")
    hour_df = pd.read_csv("dashboard/hour.csv")
    return day_df, hour_df

day_df, hour_df = load_data()

# Data Cleaning Function
def clean_bike_data(df, is_hourly=False):
    df = df.copy()
    df.drop(columns=['instant', 'workingday'], errors='ignore', inplace=True)
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    # Rename Columns
    col_map = {'yr': 'year', 'mnth': 'month', 'weekday': 'day_of_week', 'weathersit': 'weather', 'cnt': 'total_rentals'}
    if is_hourly:
        col_map['hr'] = 'hour'
    df.rename(columns=col_map, inplace=True)
    
    # Categorical Mapping
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    df['season'] = df['season'].map(season_map)

    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['month'] = df['month'].map(month_map)

    weather_map = {1: 'Clear', 2: 'Cloudy/Misty', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
    df['weather'] = df['weather'].map(weather_map)

    day_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
               4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    df['day_of_week'] = df['day_of_week'].map(day_map)

    df['year'] = df['year'].map({0: '2011', 1: '2012'})
    
    df['day_type'] = df['day_of_week'].apply(
        lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday'
    )
    
    df['temp'] = df['temp'] * 41
    
    df['month_year'] = df['dteday'].dt.to_period('M')
    
    return df

# Cleaned Data
day_clean = clean_bike_data(day_df)
hour_clean = clean_bike_data(hour_df, is_hourly=True)

# Sidebar
st.sidebar.header("Filter Data")
selected_season = st.sidebar.selectbox("Select Season", ["All"] + list(day_clean["season"].unique()))
selected_weather = st.sidebar.selectbox("Select Weather", ["All"] + list(day_clean["weather"].unique()))

# Filter Data
filtered_data = day_clean.copy()
if selected_season != "All":
    filtered_data = filtered_data[filtered_data["season"] == selected_season]
if selected_weather != "All":
    filtered_data = filtered_data[filtered_data["weather"] == selected_weather]

# Dashboard Title
st.title("🚲 Bike Sharing Data Dashboard")
st.markdown("### Explore Bike Sharing Rental Patterns 📊")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Rentals", f"{filtered_data['total_rentals'].sum():,}")
col2.metric("Average Daily Rentals", f"{filtered_data['total_rentals'].mean():,.2f}")
col3.metric("Total Days", f"{filtered_data.shape[0]}")

# Distribution of Numerical Variables
st.subheader("📊 Distribution of Numerical Variables")

numerical_cols = ['temp', 'hum', 'windspeed', 'total_rentals']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, col in enumerate(numerical_cols):
    row, col_idx = divmod(i, 2)
    sns.histplot(day_clean[col], kde=True, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'Distribution of {col}')

plt.tight_layout()
st.pyplot(fig)

# Distribution of Categorical Variables
st.subheader("📊 Distribution of Categorical Variables")

categorical_cols = ['season', 'month', 'weather', 'day_of_week']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for i, col in enumerate(categorical_cols):
    row, col_idx = divmod(i, 2)
    sns.countplot(data=day_clean, x=col, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'Distribution of {col}')
    axes[row, col_idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig)

# Weather vs Rentals
st.subheader("🌦️ Bike Rentals by Weather Condition")

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='weather', y='total_rentals', data=day_clean, ax=ax)
ax.set_title('Bike Rentals by Weather Condition')
ax.tick_params(axis='x', rotation=45)

st.pyplot(fig)

# Temperature vs Rentals (Casual vs Registered)
st.subheader("🌡️ Effect of Temperature on Different User Types")

fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x='temp', y='casual', data=day_clean, label='Casual', ax=ax)
sns.scatterplot(x='temp', y='registered', data=day_clean, label='Registered', ax=ax)
ax.set_title('Effect of Temperature on Different User Types')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Number of Rentals')
ax.legend()

st.pyplot(fig)


# Correlation Matrix
st.subheader("Correlation Matrix of Numerical Variables")
corr_matrix = day_clean[["temp", "hum", "windspeed", "casual", "registered", "total_rentals"]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Weekday vs Weekend Hourly Patterns
st.subheader("📅 Weekday vs Weekend Hourly Bike Rentals")

# Aggregation for weekday vs weekend patterns
weekday_hourly = hour_clean[hour_clean['day_type'] == 'Weekday'].groupby('hour')['total_rentals'].mean().reset_index()
weekend_hourly = hour_clean[hour_clean['day_type'] == 'Weekend'].groupby('hour')['total_rentals'].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(weekday_hourly['hour'], weekday_hourly['total_rentals'], label='Weekday', marker='o', color='#1f77b4')
ax.plot(weekend_hourly['hour'], weekend_hourly['total_rentals'], label='Weekend', marker='o', color='#ff7f0e')

ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Rentals')
ax.set_title('Average Hourly Bike Rentals: Weekday vs Weekend')
ax.set_xticks(range(24))
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)

# Seasonal Bike Rentals
st.subheader("🌦️ Total Bike Rentals by Season")

# Aggregate total rentals per season
seasonal_rentals = day_clean.groupby('season')['total_rentals'].sum().reindex(
    ['Spring', 'Summer', 'Fall', 'Winter']
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='season', y='total_rentals', data=seasonal_rentals, palette='viridis', ax=ax)
ax.set_title('Total Bike Rentals by Season')
ax.set_xlabel('Season')
ax.set_ylabel('Total Rentals')

st.pyplot(fig)

# Monthly Bike Rentals by User Type
st.subheader("📅 Monthly Bike Rentals by User Type")

# Aggregate total rentals per month for casual & registered users
user_type_monthly = day_clean.groupby(['month_year']).agg({
    'casual': 'sum',
    'registered': 'sum',
    'total_rentals': 'sum'
}).reset_index()

# Convert Period to string for better plotting
user_type_monthly['month_year_str'] = user_type_monthly['month_year'].astype(str)

fig, ax = plt.subplots(figsize=(14, 7))
width = 0.35
x = np.arange(len(user_type_monthly))

ax.bar(x - width/2, user_type_monthly['casual'], width, label='Casual', color='#ff7f0e')
ax.bar(x + width/2, user_type_monthly['registered'], width, label='Registered', color='#1f77b4')

ax.set_xlabel('Month')
ax.set_ylabel('Number of Rentals')
ax.set_title('Monthly Bike Rentals by User Type')
ax.set_xticks(x)
ax.set_xticklabels(user_type_monthly['month_year_str'], rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# Footer Section
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <h3>🚴 Bike Sharing Data Dashboard</h3>
        <p>Made with ❤️ by <b>Evan Arlen Handy</b></p>
        <p>📧 <a href="mailto:cloaaa00@gmail.com" style="text-decoration:none;">cloaaa00@gmail.com</a></p>
        <p>🔗 <a href="https://www.dicoding.com/users/warlord194" target="_blank" style="text-decoration:none;">My Dicoding Profile</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
