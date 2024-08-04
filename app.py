import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# API keys
geocoding_api_key = '80843f03ed6b4945a45f1bd8c51e5c2f'
weather_api_key = 'b53305cd6b960c1984aed0acaf76aa2e'

# Function to get latitude and longitude from village name
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

# Function to get weather forecast
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

# Load and preprocess fertilizer recommendation data
fertilizer_url = 'https://raw.githubusercontent.com/dheerajreddy71/Design_Project/main/fertilizer_recommendation.csv'
data = pd.read_csv(fertilizer_url)
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

# Train a Random Forest Classifier for fertilizer recommendation
fertilizer_model = RandomForestClassifier()
fertilizer_model.fit(x_train, y_train)

# Load and prepare datasets for yield prediction and crop recommendation
yield_df = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/yield_df.csv")
crop_recommendation_data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/Crop_recommendation.csv")

# Preprocessing for yield prediction
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

# Preprocessing for crop recommendation
crop_X = crop_recommendation_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
crop_y = crop_recommendation_data['label']
crop_X_train, crop_X_test, crop_y_train, crop_y_test = train_test_split(crop_X, crop_y, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(crop_X_train, crop_y_train)

# Load crop data for temperature and humidity prediction
temp_humidity_data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/ds1.csv", encoding='ISO-8859-1')
temp_humidity_data = temp_humidity_data.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
X = temp_humidity_data.drop(['Crop', 'Temperature Required (°F)'], axis=1)
y = temp_humidity_data['Temperature Required (°F)']
temp_humidity_model = LinearRegression()
temp_humidity_model.fit(X, y)

# Load pest warnings data
pest_data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/ds2.csv")
crop_pest_data = {}
planting_time_info = {}
growth_stage_info = {}
pesticides_info = {}

for _, row in pest_data.iterrows():
    crop = row[0].strip().lower()
    pest = row[1].strip()
    crop_pest_data[crop] = pest
    planting_time_info[crop] = row[5].strip()
    growth_stage_info[crop] = row[6].strip()
    pesticides_info[crop] = row[4].strip()

def predict_requirements(crop_name):
    crop_name = crop_name.lower()
    crop_data = temp_humidity_data[temp_humidity_data['Crop'].str.lower() == crop_name].drop(['Crop', 'Temperature Required (°F)'], axis=1)
    if crop_data.empty:
        return None, None
    predicted_temperature = temp_humidity_model.predict(crop_data)
    crop_row = temp_humidity_data[temp_humidity_data['Crop'].str.lower() == crop_name]
    humidity_required = crop_row['Humidity Required (%)'].values[0]
    return humidity_required, predicted_temperature[0]

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

price_model = RandomForestRegressor(n_estimators=100, random_state=42)
price_model.fit(price_X_train, price_y_train)

# Streamlit app
st.title("Smart Agriculture Assistant")

option = st.selectbox("Select Functionality", ["Predict Fertilizer Requirement", "Predict Crop Yield", "Recommend Crops", "Weather Forecast", "Predict Pests", "Crop Prices"])

if option == "Predict Fertilizer Requirement":
    st.header("Fertilizer Recommendation")
    st.write("Enter soil type, crop type, and other factors to get fertilizer recommendations.")
    soil_type = st.selectbox("Soil Type", data['Soil_Type'].unique())
    crop_type = st.selectbox("Crop Type", data['Crop_Type'].unique())
    nitrogen = st.number_input("Nitrogen (N) in soil", min_value=0.0)
    phosphorus = st.number_input("Phosphorus (P) in soil", min_value=0.0)
    potassium = st.number_input("Potassium (K) in soil", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0)

    input_features = np.array([[soil_type, crop_type, nitrogen, phosphorus, potassium, temperature, humidity]])
    predicted_fertilizer = fertilizer_model.predict(input_features)
    fertilizer_name = encode_ferti.inverse_transform(predicted_fertilizer)
    st.write(f"Recommended Fertilizer: {fertilizer_name[0]}")

elif option == "Predict Crop Yield":
    st.header("Crop Yield Prediction")
    st.write("Enter the parameters to predict the crop yield.")
    year = st.number_input("Year", min_value=2000)
    rainfall = st.number_input("Rainfall (mm/year)", min_value=0.0)
    pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0)
    avg_temp = st.number_input("Average Temperature (°C)", min_value=0.0)
    area = st.number_input("Area (hectares)", min_value=0.0)
    crop_item = st.selectbox("Crop Item", temp_humidity_data['Crop'].unique())

    input_data = np.array([[year, rainfall, pesticides, avg_temp, area, crop_item]])
    input_data_preprocessed = yield_preprocessor.transform(input_data)
    predicted_yield = yield_model.predict(input_data_preprocessed)
    st.write(f"Predicted Crop Yield: {predicted_yield[0]}")

elif option == "Recommend Crops":
    st.header("Crop Recommendation")
    st.write("Enter soil and climate parameters to get crop recommendations.")
    nitrogen = st.number_input("Nitrogen (N) in soil", min_value=0.0)
    phosphorus = st.number_input("Phosphorus (P) in soil", min_value=0.0)
    potassium = st.number_input("Potassium (K) in soil", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("Soil pH", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm/year)", min_value=0.0)

    input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    recommended_crop = crop_model.predict(input_features)
    crop_name = encode_crop.inverse_transform(recommended_crop)
    st.write(f"Recommended Crop: {crop_name[0]}")

elif option == "Weather Forecast":
    st.header("Weather Forecast")
    village_name = st.text_input("Enter Village Name")
    if village_name:
        latitude, longitude = get_lat_lon(village_name)
        if latitude and longitude:
            forecast = get_weather_forecast(latitude, longitude)
            if forecast:
                st.write("Weather Forecast:")
                for item in forecast:
                    st.write(f"{item['date']} {item['time']}: Temp: {item['temp']}°C, Pressure: {item['pressure']} hPa, Humidity: {item['humidity']}%, Weather: {item['weather']}")
            else:
                st.write("No weather forecast data available.")
        else:
            st.write("Location not found.")

elif option == "Predict Pests":
    st.header("Pest Prediction")
    crop_name = st.text_input("Enter Crop Name")
    if crop_name:
        pest_warning = predict_pest_warnings(crop_name)
        if pest_warning:
            st.write("Pest Warnings:")
            st.write(pest_warning)
        else:
            st.write("No pest information available for this crop.")

elif option == "Crop Prices":
    st.header("Crop Price Prediction")
    st.write("Enter the parameters to predict crop prices.")
    state = st.selectbox("State", price_data['state'].unique())
    district = st.selectbox("District", price_data['district'].unique())
    market = st.selectbox("Market", price_data['market'].unique())
    commodity = st.selectbox("Commodity", price_data['commodity'].unique())
    variety = st.selectbox("Variety", price_data['variety'].unique())
    day = st.number_input("Day", min_value=1, max_value=31)
    month = st.number_input("Month", min_value=1, max_value=12)
    year = st.number_input("Year", min_value=2000)
    
    input_features = pd.DataFrame({
        'state': [state],
        'district': [district],
        'market': [market],
        'commodity': [commodity],
        'variety': [variety],
        'day': [day],
        'month': [month],
        'year': [year]
    })
    
    input_encoded = price_encoder.transform(input_features)
    predicted_prices = price_model.predict(input_encoded)
    st.write(f"Predicted Prices - Min: {predicted_prices[0][0]}, Max: {predicted_prices[0][1]}, Modal: {predicted_prices[0][2]}")
