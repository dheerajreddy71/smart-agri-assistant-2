import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = pd.read_csv('C:/Users/Windows/Downloads/fertilizer_recommendation.csv')
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
st.title("Fertilizer Recommendation System")

# Input fields
temperature = st.number_input('Temperature', format="%.2f")
humidity = st.number_input('Humidity', format="%.2f")
moisture = st.number_input('Moisture', format="%.2f")
soil_type = st.selectbox('Soil Type', encode_soil.classes_)
crop_type = st.selectbox('Crop Type', encode_crop.classes_)
nitrogen = st.number_input('Nitrogen', format="%.2f")
potassium = st.number_input('Potassium', format="%.2f")
phosphorous = st.number_input('Phosphorous', format="%.2f")

if st.button('Predict'):
    try:
        # Encode the soil type and crop type
        soil_type_encoded = encode_soil.transform([soil_type])[0]
        crop_type_encoded = encode_crop.transform([crop_type])[0]

        # Make predictions
        prediction = rand.predict([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]])

        # Decode the predicted fertilizer
        recommended_fertilizer = encode_ferti.inverse_transform(prediction)[0]
        st.write(f"Recommended Fertilizer: {recommended_fertilizer}")

    except Exception as e:
        st.error(f"Error: {e}")
