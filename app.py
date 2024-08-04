import streamlit as st
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# Streamlit app
st.markdown("""
    <style>
        .reportview-container {
            background: url("https://github.com/dheerajreddy71/Webbuild/blob/main/background.jpg") no-repeat center center fixed;
            background-size: cover;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Weather Forecast and Fertilizer Recommendation System")

# Weather Forecast Section
st.header("Weather Forecast for Village")
village_name = st.text_input('Enter village name')

if st.button('Fetch Weather'):
    if village_name:
        latitude, longitude = get_lat_lon(village_name)
        if latitude and longitude:
            st.write(f'Coordinates: Latitude {latitude}, Longitude {longitude}')
            forecast = get_weather_forecast(latitude, longitude)
            if forecast:
                df = pd.DataFrame(forecast)
                st.write('Weather Forecast:')
                st.dataframe(df)
            else:
                st.write('Weather forecast data not available.')
        else:
            st.write('Village not found.')
    else:
        st.write('Please enter a village name.')

# Fertilizer Recommendation Section
st.header("Fertilizer Recommendation System")

# Input fields for fertilizer recommendation
temperature = st.number_input('Temperature', format="%.2f")
humidity = st.number_input('Humidity', format="%.2f")
moisture = st.number_input('Moisture', format="%.2f")
soil_type = st.selectbox('Soil Type', encode_soil.classes_)
crop_type = st.selectbox('Crop Type', encode_crop.classes_)
nitrogen = st.number_input('Nitrogen', format="%.2f")
potassium = st.number_input('Potassium', format="%.2f")
phosphorous = st.number_input('Phosphorous', format="%.2f")

if st.button('Predict Fertilizer'):
    try:
        soil_type_encoded = encode_soil.transform([soil_type])[0]
        crop_type_encoded = encode_crop.transform([crop_type])[0]
        prediction = rand.predict([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]])
        recommended_fertilizer = encode_ferti.inverse_transform(prediction)[0]
        st.write(f"Recommended Fertilizer: {recommended_fertilizer}")
    except Exception as e:
        st.error(f"Error: {e}")
