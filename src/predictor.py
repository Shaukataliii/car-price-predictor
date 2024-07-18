import pickle, os, sklearn, streamlit
import pandas as pd
import numpy as np

@streamlit.cache_data
def load_cache_resource():
    predictor = Predictor()
    return predictor

class Predictor:
    def __init__(self):
        self.model_pipeline = self.load_model()
        self.car_names = self.model_pipeline.steps[0][1].transformers[0][1].categories[0]
        self.companies = self.model_pipeline.steps[0][1].transformers[0][1].categories[1]
        self.fuel_types = self.model_pipeline.steps[0][1].transformers[0][1].categories[2]

    def load_model(self):
        cwd = os.getcwd()
        model_filepath = os.path.join(cwd, "src", "resources", "model_pipeline.pkl")
        return pickle.load(open(model_filepath, "rb"))

    def predict_price(self, user_input: list):
        user_input = self.transform_input(user_input)
        prediction = self.model_pipeline.predict(user_input)[0]
        prediction = np.round(prediction, 0)
        return prediction

    def transform_input(self, user_input: list):
        return pd.DataFrame([user_input], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

