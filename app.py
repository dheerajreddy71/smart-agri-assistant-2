import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# API keys
geocoding_api_key = '80843f03ed6b4945a45f1bd8c51e5c2f'
weather_api_key = 'b53305cd6b960c1984aed0acaf76aa2e'

def get_lat_lon(village_name):
    geocoding_url = f'https://api.opencagedata.com/geocode/v1/json?q={village_name}&key={geocoding_api_key}'
    try:
        response = requests.get(geocoding_url)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            latitude = data['results'][0]['geometry']['lat']
            longitude = data['results'][0]['geometry']['lng']
            return latitude, longitude
        else:
            return None, None
    except Exception as e:
        st.error(f"Error fetching geocoding data: {e}")
        return None, None

def get_weather_forecast(latitude, longitude):
    weather_url = f'https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&units=metric&cnt=40&appid={weather_api_key}'
    try:
        response = requests.get(weather_url)
        response.raise_for_status()
        data = response.json()
        if data['cod'] == '200':
            forecast = []
            for item in data['list']:
                date_time = item['dt_txt']
                date, time = date_time.split(' ')
                forecast.append({
                    'date': date,
                    'time': time,
                    'temp': item['main']['temp'],
                    'pressure': item['main']['pressure'],
                    'humidity': item['main']['humidity'],
                    'weather': item['weather'][0]['description']
                })
            return forecast
        else:
            st.error(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

# Load and preprocess data for fertilizer recommendation
url = 'https://raw.githubusercontent.com/dheerajreddy71/Design_Project/main/fertilizer_recommendation.csv'
data = pd.read_csv(url)
data.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'}, inplace=True)
data.dropna(inplace=True)

# Encode categorical variables
encode_soil = LabelEncoder()
data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)

encode_crop = LabelEncoder()
data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)

encode_ferti = LabelEncoder()
data.Fertilizer = encode_ferti.fit_transform(data.Fertilizer)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data.drop('Fertilizer', axis=1), data.Fertilizer, test_size=0.2, random_state=1)

# Train a Random Forest Classifier
rand = RandomForestClassifier()
rand.fit(x_train, y_train)

# Load and prepare datasets for yield prediction
yield_df = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/yield_df.csv")
crop_recommendation_data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/Crop_recommendation.csv")

yield_preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), [0, 1, 2, 3]),
        ('OHE', OneHotEncoder(drop='first'), [4, 5]),
    ],
    remainder='passthrough'
)
yield_X = yield_df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item']]
yield_y = yield_df['hg/ha_yield']
yield_X_train, yield_X_test, yield_y_train, yield_y_test = train_test_split(yield_X, yield_y, train_size=0.8, random_state=0, shuffle=True)
yield_X_train_dummy = yield_preprocessor.fit_transform(yield_X_train)
yield_X_test_dummy = yield_preprocessor.transform(yield_X_test)
yield_model = KNeighborsRegressor(n_neighbors=5)
yield_model.fit(yield_X_train_dummy, yield_y_train)

crop_X = crop_recommendation_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
crop_y = crop_recommendation_data['label']
crop_X_train, crop_X_test, crop_y_train, crop_y_test = train_test_split(crop_X, crop_y, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(crop_X_train, crop_y_train)

# Load crop data and train the model for temperature prediction
data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/ds1.csv", encoding='ISO-8859-1')
data = data.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
X = data.drop(['Crop', 'Temperature Required (°F)'], axis=1)
y = data['Temperature Required (°F)']
model = LinearRegression()
model.fit(X, y)

# Function to predict temperature and humidity requirements for a crop
def predict_requirements(crop_name):
    crop_name = crop_name.lower()
    crop_data = data[data['Crop'].str.lower() == crop_name].drop(['Crop', 'Temperature Required (°F)'], axis=1)
    if crop_data.empty:
        return None, None  # Handle cases where crop_name is not found
    predicted_temperature = model.predict(crop_data)
    crop_row = data[data['Crop'].str.lower() == crop_name]
    humidity_required = crop_row['Humidity Required (%)'].values[0]
    return humidity_required, predicted_temperature[0]

# Function to get pest warnings for a crop
crop_pest_data = {}
planting_time_info = {}
growth_stage_info = {}
pesticides_info = {}

# Read data from the CSV file and store it in dictionaries
pest_data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/ds2.csv")
for _, row in pest_data.iterrows():
    crop = row[0].strip().lower()
    pest = row[1].strip()
    crop_pest_data[crop] = pest
    planting_time_info[crop] = row[5].strip()
    growth_stage_info[crop] = row[6].strip()
    pesticides_info[crop] = row[4].strip()

def predict_pest_warnings(crop_name):
    crop_name = crop_name.lower()
    specified_crops = [crop_name]

    pest_warnings = []

    for crop in specified_crops:
        if crop in crop_pest_data:
            pests = crop_pest_data[crop].split(', ')
            warning_message = f"\nBeware of pests like {', '.join(pests)} for {crop.capitalize()}.\n"

            if crop in planting_time_info:
                planting_time = planting_time_info[crop]
                warning_message += f"\nPlanting Time: {planting_time}\n"

            if crop in growth_stage_info:
                growth_stage = growth_stage_info[crop]
                warning_message += f"\nGrowth Stages of Plant: {growth_stage}\n"

            if crop in pesticides_info:
                pesticides = pesticides_info[crop]
                warning_message += f"\nUse Pesticides like: {pesticides}\n"
                
            pest_warnings.append(warning_message)

    return '\n'.join(pest_warnings)

# Load and preprocess crop price data
price_data = pd.read_csv('https://github.com/dheerajreddy71/Design_Project/raw/main/pred_data.csv', encoding='ISO-8859-1')
price_data['arrival_date'] = pd.to_datetime(price_data['arrival_date'])
price_data['day'] = price_data['arrival_date'].dt.day
price_data['month'] = price_data['arrival_date'].dt.month
price_data['year'] = price_data['arrival_date'].dt.year
price_data.drop(['arrival_date'], axis=1, inplace=True)

price_X = price_data.drop(['min_price', 'max_price', 'modal_price'], axis=1)
price_y = price_data[['min_price', 'max_price', 'modal_price']]

price_encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['state', 'district', 'market', 'commodity', 'variety'])
    ],
    remainder='passthrough'
)

price_X_encoded = price_encoder.fit_transform(price_X)
price_X_train, price_X_test, price_y_train, price_y_test = train_test_split(price_X_encoded, price_y, test_size=0.2, random_state=42)

price_model = LinearRegression()
price_model.fit(price_X_train, price_y_train)

# Streamlit UI
st.title("Smart Agriculture Assistant")

# Weather Forecast
st.header("Weather Forecast")
village_name = st.text_input("Enter your village name:")
if village_name:
    lat, lon = get_lat_lon(village_name)
    if lat and lon:
        weather_data = get_weather_forecast(lat, lon)
        if weather_data:
            df_weather = pd.DataFrame(weather_data)
            st.write(df_weather)

# Fertilizer Recommendation
st.header("Fertilizer Recommendation")
soil_type = st.selectbox("Select Soil Type", options=data['Soil_Type'].unique())
humidity = st.slider("Select Humidity", min_value=int(data['Humidity'].min()), max_value=int(data['Humidity'].max()), value=int(data['Humidity'].median()))
crop_type = st.selectbox("Select Crop Type", options=data['Crop_Type'].unique())

if st.button("Get Fertilizer Recommendation"):
    soil_type_encoded = encode_soil.transform([soil_type])[0]
    crop_type_encoded = encode_crop.transform([crop_type])[0]
    recommendation = rand.predict([[soil_type_encoded, humidity, crop_type_encoded]])
    st.write(f"Recommended Fertilizer: {encode_ferti.inverse_transform(recommendation)[0]}")

# Crop Yield Prediction
st.header("Crop Yield Prediction")
rainfall = st.number_input("Enter Rainfall (mm)", min_value=0.0)
pesticides = st.number_input("Enter Pesticides (tonnes)", min_value=0.0)
avg_temp = st.number_input("Enter Average Temperature (°C)", min_value=-50.0, max_value=50.0)
area = st.number_input("Enter Area (hectares)", min_value=0.0)

if st.button("Predict Yield"):
    prediction = yield_model.predict(yield_preprocessor.transform([[2024, rainfall, pesticides, avg_temp, area, 'maize']]))  # Example crop 'maize'
    st.write(f"Predicted Yield: {prediction[0]} hg/ha")

# Crop Recommendation
st.header("Crop Recommendation")
N = st.number_input("Enter Nitrogen (N) level", min_value=0.0)
P = st.number_input("Enter Phosphorus (P) level", min_value=0.0)
K = st.number_input("Enter Potassium (K) level", min_value=0.0)
temperature = st.number_input("Enter Temperature (°C)", min_value=-50.0, max_value=50.0)
humidity = st.number_input("Enter Humidity (%)", min_value=0, max_value=100)
ph = st.number_input("Enter Soil pH", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Enter Rainfall (mm)", min_value=0.0)

if st.button("Get Crop Recommendation"):
    crop_prediction = crop_model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    st.write(f"Recommended Crop: {crop_prediction[0]}")

# Crop Requirements and Pest Warnings
st.header("Crop Requirements and Pest Warnings")
crop_name = st.text_input("Enter Crop Name for Requirements and Pest Warnings:")
if crop_name:
    humidity_required, temp_required = predict_requirements(crop_name)
    if humidity_required is not None:
        st.write(f"Humidity Required: {humidity_required}%")
        st.write(f"Temperature Required: {temp_required}°F")

    pest_warnings = predict_pest_warnings(crop_name)
    if pest_warnings:
        st.write(pest_warnings)

# Crop Price Prediction
st.header("Crop Price Prediction")
state = st.selectbox("Select State", options=price_data['state'].unique())
district = st.selectbox("Select District", options=price_data['district'].unique())
market = st.selectbox("Select Market", options=price_data['market'].unique())
commodity = st.selectbox("Select Commodity", options=price_data['commodity'].unique())
variety = st.selectbox("Select Variety", options=price_data['variety'].unique())
date = st.date_input("Select Date")

if st.button("Predict Price"):
    day = date.day
    month = date.month
    year = date.year
    encoded_price_X = price_encoder.transform([[state, district, market, commodity, variety, day, month, year]])
    min_price, max_price, modal_price = price_model.predict(encoded_price_X)[0]
    st.write(f"Predicted Prices - Min: {min_price}, Max: {max_price}, Modal: {modal_price}")
