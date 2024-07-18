import streamlit as st
from src.predictor import load_cache_resource

st.set_page_config("Car Price Predictor", page_icon="racing_car")
resource = load_cache_resource()

st.title(":racing_car: Car Price Predictor")
st.caption("The model has a r2_score of 81.5.")
car_name = st.selectbox("Choose car model", options = resource.car_names)
company = st.selectbox("Choose company name", options = resource.companies)
year = st.number_input("Year car purchased.")
distance_travelled = st.number_input("Distance travelled (kms)")
fuel_type = st.selectbox("Choose Fuel type", options = resource.fuel_types)
submit = st.button("Predict Price", type='primary')

if submit:
    with st.spinner("Predicting.."):
        prediction = resource.predict_price([car_name, company, year, distance_travelled, fuel_type])
        st.write(":blue[Predicted Price]:", prediction)
