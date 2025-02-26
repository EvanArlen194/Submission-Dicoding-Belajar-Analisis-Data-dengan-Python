import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import plotly.express as px

def clean_bike_data(df, is_hourly=False):
    """
    Clean and transform bike sharing dataset.
    
    Parameters:
    df (DataFrame): DataFrame to clean.
    is_hourly (bool): Whether the dataset contains hourly data.
    
    Returns:
    DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if 'instant' in cleaned_df.columns:
        cleaned_df.drop('instant', axis=1, inplace=True)
    if 'workingday' in cleaned_df.columns:
        cleaned_df.drop('workingday', axis=1, inplace=True)
    
    cleaned_df['dteday'] = pd.to_datetime(cleaned_df['dteday'])
    
    cat_columns = ['season', 'mnth', 'holiday', 'weekday', 'weathersit']
    for col in cat_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype('category')
    
    column_mapping = {
        'yr': 'year',
        'mnth': 'month',
        'weekday': 'day_of_week', 
        'weathersit': 'weather',
        'windspeed': 'wind_speed',
        'cnt': 'total_rentals',
        'hum': 'humidity',
    }
    if is_hourly:
        column_mapping['hr'] = 'hour'
    cleaned_df.rename(columns=column_mapping, inplace=True)

    season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    cleaned_df['season'] = cleaned_df['season'].map(season_mapping)
    
    month_mapping = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    cleaned_df['month'] = cleaned_df['month'].map(month_mapping)
    
    weather_mapping = {
        1: 'Clear', 2: 'Cloudy/Misty', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'
    }
    cleaned_df['weather'] = cleaned_df['weather'].map(weather_mapping)
    
    day_mapping = {
        0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
        4: 'Thursday', 5: 'Friday', 6: 'Saturday'
    }
    cleaned_df['day_of_week'] = cleaned_df['day_of_week'].map(day_mapping)
    
    cleaned_df['year'] = cleaned_df['year'].map({0: '2011', 1: '2012'})
    
    cleaned_df['humidity'] = cleaned_df['humidity'] * 100
    
    cleaned_df['day_type'] = cleaned_df['day_of_week'].apply(
        lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday'
    )
    
    cleaned_df['temp_celsius'] = cleaned_df['temp'] * 41
    
    return cleaned_df

def main():
    st.title("🚴‍♂️ Bike Sharing Data Analysis 🚴‍♀️")
    st.markdown("""
    **Selamat datang di aplikasi analisis data penyewaan sepeda!**  
    Aplikasi ini menampilkan berbagai visualisasi untuk memahami pola penyewaan sepeda berdasarkan waktu, cuaca, dan faktor lainnya.
    """)
    
    day_df = pd.read_csv("day.csv")
    hour_df = pd.read_csv("hour.csv")
    
    day_clean = clean_bike_data(day_df)
    hour_clean = clean_bike_data(hour_df, is_hourly=True)
    
    st.sidebar.header("📊 Dataset Preview")
    if st.sidebar.checkbox("Tampilkan dataset"):
        st.write("### Dataset yang Telah Dibersihkan")
        st.dataframe(day_clean.head())
    
    # Visualisasi 1: Pola Penyewaan per Jam (Weekday vs Weekend)
    st.header("⏰ Pola Penyewaan per Jam (Weekday vs Weekend)")
    st.markdown("""
    Berikut adalah pola penyewaan sepeda per jam pada hari kerja (Weekday) dan akhir pekan (Weekend).
    """)
    weekday_hourly = hour_clean[hour_clean['day_type'] == 'Weekday'].groupby('hour')['total_rentals'].mean().reset_index()
    weekend_hourly = hour_clean[hour_clean['day_type'] == 'Weekend'].groupby('hour')['total_rentals'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(weekday_hourly['hour'], weekday_hourly['total_rentals'], label='Weekday', marker='o', color='#1f77b4')
    ax.plot(weekend_hourly['hour'], weekend_hourly['total_rentals'], label='Weekend', marker='o', color='#ff7f0e')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Rentals')
    ax.set_title('Average Hourly Bike Rentals: Weekday vs Weekend')
    ax.legend()
    st.pyplot(fig)
    
    # Visualisasi 2: Penyewaan Harian berdasarkan Hari dalam Seminggu
    st.header("📅 Penyewaan Harian berdasarkan Hari dalam Seminggu")
    st.markdown("""
    Berikut adalah rata-rata penyewaan sepeda berdasarkan hari dalam seminggu.
    """)
    daily_rentals = day_clean.groupby('day_of_week')['total_rentals'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='day_of_week', y='total_rentals', data=daily_rentals, palette='viridis', ax=ax)
    ax.set_title('Average Daily Bike Rentals by Day of Week')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Rentals')
    st.pyplot(fig)
    
    # Visualisasi 3: Penyewaan berdasarkan Musim
    st.header("🌦️ Penyewaan berdasarkan Musim")
    st.markdown("""
    Berikut adalah total penyewaan sepeda berdasarkan musim.
    """)
    seasonal_rentals = day_clean.groupby('season')['total_rentals'].sum().reindex(
        ['Spring', 'Summer', 'Fall', 'Winter']
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='season', y='total_rentals', data=seasonal_rentals, palette='viridis', ax=ax)
    ax.set_title('Total Bike Rentals by Season')
    ax.set_xlabel('Season')
    ax.set_ylabel('Total Rentals')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    st.pyplot(fig)
    
    # Visualisasi 4: Distribusi Tipe Pengguna (Casual vs Registered)
    st.header("👥 Distribusi Tipe Pengguna (Casual vs Registered)")
    st.markdown("""
    Berikut adalah distribusi penyewaan sepeda berdasarkan tipe pengguna (Casual vs Registered).
    """)
    total_casual = day_clean['casual'].sum()
    total_registered = day_clean['registered'].sum()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie([total_casual, total_registered], labels=['Casual', 'Registered'], autopct='%1.1f%%', 
           colors=['#ff7f0e', '#1f77b4'], explode=(0.05, 0), startangle=90, shadow=True)
    ax.set_title('Distribution of Rental Types')
    st.pyplot(fig)
    
    # Visualisasi 5: Pengaruh Cuaca terhadap Penyewaan
    st.header("☀️ Pengaruh Cuaca terhadap Penyewaan")
    st.markdown("""
    Berikut adalah rata-rata penyewaan sepeda berdasarkan kondisi cuaca.
    """)
    weather_impact = day_clean.groupby('weather')['total_rentals'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='weather', y='total_rentals', data=weather_impact, palette='viridis', ax=ax)
    ax.set_title('Average Daily Rentals by Weather Condition')
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Average Rentals')
    st.pyplot(fig)
    
    # Visualisasi 6: Hubungan Suhu dan Penyewaan
    st.header("🌡️ Hubungan Suhu dan Penyewaan")
    st.markdown("""
    Berikut adalah hubungan antara suhu (°C) dan jumlah penyewaan sepeda.
    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='temp_celsius', y='total_rentals', data=day_clean, scatter_kws={'alpha':0.5}, 
                line_kws={'color': 'red'}, ax=ax)
    ax.set_title('Relationship Between Temperature and Bike Rentals')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Daily Rentals')
    st.pyplot(fig)

if __name__ == "__main__":
    main()