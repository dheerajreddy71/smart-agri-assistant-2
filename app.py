import streamlit as st
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from googletrans import Translator

# API keys
geocoding_api_key = '80843f03ed6b4945a45f1bd8c51e5c2f'
weather_api_key = 'b53305cd6b960c1984aed0acaf76aa2e'

# Translator setup
translator = Translator()

# Function to translate text based on selected language
def translate_text(text, dest_lang):
    try:
        translation = translator.translate(text, dest=dest_lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Function to get geolocation
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
def get_weather_forecast(latitude, longitude, dest_lang):
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
                weather_desc = translate_text(item['weather'][0]['description'], dest_lang)
                forecast.append({
                    'date': date,
                    'time': time,
                    'temp': item['main']['temp'],
                    'pressure': item['main']['pressure'],
                    'humidity': item['main']['humidity'],
                    'weather': weather_desc
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
st.set_page_config(page_title="Smart Agri Assistant", layout="wide", page_icon="🌾")

# Add a background image
page_bg_img = '''
<style>
.stApp {
background-image: url("https://github.com/dheerajreddy71/Webbuild/raw/main/background.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Language selection
st.sidebar.header("Language Selection")
languages = {'English': 'en', 'Hindi': 'hi', 'Tamil': 'ta', 'Telugu': 'te'}
selected_language = st.sidebar.selectbox('Select language', list(languages.keys()))

# Get the selected language code
dest_lang = languages[selected_language]

# Weather Forecast Section
st.header(translate_text("Weather Forecast for Village", dest_lang))
village_name = st.text_input(translate_text('Enter village name', dest_lang))

if st.button(translate_text('Fetch Weather', dest_lang)):
    if village_name:
        latitude, longitude = get_lat_lon(village_name)
        if latitude and longitude:
            st.write(translate_text(f'Coordinates: Latitude {latitude}, Longitude {longitude}', dest_lang))
            forecast = get_weather_forecast(latitude, longitude, dest_lang)
            if forecast:
                df = pd.DataFrame(forecast)
                st.write(translate_text('Weather Forecast:', dest_lang))
                st.dataframe(df)
            else:
                st.write(translate_text('Weather forecast data not available.', dest_lang))
        else:
            st.write(translate_text('Village not found.', dest_lang))
    else:
        st.write(translate_text('Please enter a village name.', dest_lang))

# Fertilizer Recommendation Section
st.header(translate_text("Fertilizer Recommendation System", dest_lang))

# Input fields for fertilizer recommendation
temperature = st.number_input(translate_text('Temperature', dest_lang), format="%.2f")
humidity = st.number_input(translate_text('Humidity', dest_lang), format="%.2f")
moisture = st.number_input(translate_text('Moisture', dest_lang), format="%.2f")
soil_type = st.selectbox(translate_text('Soil Type', dest_lang), encode_soil.classes_)
crop_type = st.selectbox(translate_text('Crop Type', dest_lang), encode_crop.classes_)
nitrogen = st.number_input(translate_text('Nitrogen', dest_lang), format="%.2f")
potassium = st.number_input(translate_text('Potassium', dest_lang), format="%.2f")
phosphorous = st.number_input(translate_text('Phosphorous', dest_lang), format="%.2f")

if st.button(translate_text('Predict Fertilizer', dest_lang)):
    try:
        soil_type_encoded = encode_soil.transform([soil_type])[0]
        crop_type_encoded = encode_crop.transform([crop_type])[0]
        prediction = rand.predict([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]])
        recommended_fertilizer = encode_ferti.inverse_transform(prediction)[0]
        st.write(translate_text(f"Recommended Fertilizer: {recommended_fertilizer}", dest_lang))
    except Exception as e:
        st.error(translate_text(f"Error: {e}", dest_lang))
