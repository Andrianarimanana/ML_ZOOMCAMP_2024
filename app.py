import streamlit as st
import requests
import base64

import pickle
import pandas as pd


# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
    
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_image}");
        background-size: cover;
    }}
    h1, p, .stTextInput, .stNumberInput, .stSelectbox, .stButton {{
        color: #000000;  /* Set font color for inputs */
    }}
    
    .stButton button {{
        background-color: #4CAF50;  /* Button background color */
        color: white;  /* Button text color */
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }}
    
    .stButton button:hover {{
        background-color: #45a049;  /* Button hover color */
    }}
    </style>
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the model and utilities 

model_file = 'model_XGBClassifier.bin'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in) 
# Set background image
set_background("Image/Young-Myocarditis-Heart-Concept.webp")

st.title("Heart Disease Classification")

# Define layout with two columns
col1, col2 , col3= st.columns(3)

with col1:
    thal = st.number_input("thal", min_value=0, max_value=100, value=2)
    # slope = st.selectbox("slope", ["Male", "Female"])
    slope = st.number_input("thal", min_value=0, max_value=100, value=1)
    fbs = st.selectbox("fbs", [0,1])
    exang = st.selectbox("exang", [0,1])
    restecg = st.number_input("restecg", min_value=0, max_value=10, value=1)
    
with col2:
    id = st.number_input("id", min_value=50, max_value=300, value=224)
    age = st.number_input("age", min_value=0, max_value=200, value=53)
    sex = st.selectbox("sex", [0,1])
    dataset = st.number_input("dataset", min_value=0, max_value=10, value=0)
    cp = st.number_input("cp", min_value=0, max_value=100, value=0)
    
with col3:
    trestbps = st.number_input("trestbps", min_value=50.0, max_value=200.0, value=123.0)
    chol = st.number_input("chol", min_value=100.0, max_value=300.0, value=282.0)
    thalch = st.number_input("thalch", min_value=1.0, max_value=120.0, value=95.0)
    oldpeak = st.number_input("oldpeak", min_value=1.0, max_value=50.0, value=2.0)
    ca = st.number_input("ca", min_value=1.0, max_value=50.0, value=2.0)


# Button to make prediction
if st.button("Predict Performance"):
    input_data = {
            "thal":thal,
            "slope":slope,
            "fbs":fbs,
            "exang":exang,
            "restecg":restecg,
            "id":id,
            "age":age,
            "sex":sex,
            "dataset":dataset,
            "cp":cp,
            "trestbps":trestbps,
            "chol":chol,
            "thalch":thalch,
            "oldpeak":oldpeak,
            "ca":ca
    }
    try:
        # Create dataframe based on JSON object
        df = pd.DataFrame(input_data, index = [0])       

        # rename data columns
     
        new_col_names = ["thal","slope","fbs","exang","restecg","id","age","sex","dataset","cp","trestbps","chol","thalch","oldpeak","ca"]
        df.columns = new_col_names

        # Make a prediction
        y_pred = model.predict(df)
        y_pred_class=int(y_pred[0])
        match = {0: 'no heart disease', 1: 'Mild Heart Disease types', 2: 'Moderate Heart Disease type', 3: 'Heart Disease type', 4: 'Critical Heart Disease type'}
        y_pred_class = match[y_pred_class]
        st.success(f"Hear Prediction : {y_pred_class}")
        
    except Exception as e:
        
        st.error("Error in prediction.")