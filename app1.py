from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:5174",  
    "http://127.0.0.1:5174", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


try:
    rf_classifier = joblib.load('sid.pkl') 
    label_encoders = {
        'Water_Table': joblib.load('Water_Table_label_encoder.pkl'),
        'urbanization': joblib.load('urbanization_label_encoder.pkl'),
    }
    scaler = joblib.load('scaler.pkl') 
    logger.info(f"Scaler trained with {scaler.n_features_in_} features.")
except FileNotFoundError:
    logger.error("One or more required files are missing.")
    raise HTTPException(status_code=500, detail="Required files missing.")
except Exception as e:
    logger.error(f"Error loading files: {e}")
    raise HTTPException(status_code=500, detail="Error loading files.")

class WeatherData(BaseModel):
    Water_Table: str
    urbanization: str
    Elevation: int
    precipitation: float
    runoff_coefficient: float
    drainage: int

    @model_validator(mode='before')
    def check_negative_values(cls, values):
        for field, value in values.items():
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{field} must be non-negative.")
        return values

    @model_validator(mode='before')
    def check_empty_fields(cls, values):
        if not values.get('Water_Table') or not values.get('urbanization'):
            raise ValueError("Water_Table and urbanization cannot be empty.")
        return values

@app.post("/predict")
def predict(data: WeatherData):
    logger.info(f"Received input data: {data.dict()}")
    try:
        data_dict = data.dict()
        data_dict['Water_Table'] = label_encoders['Water_Table'].transform([data_dict['Water_Table']])[0]
        data_dict['urbanization'] = label_encoders['urbanization'].transform([data_dict['urbanization']])[0]

        input_data = pd.DataFrame([[data_dict['Water_Table'], data_dict['urbanization'],
                                    data_dict['Elevation'], data_dict['precipitation'],
                                    data_dict['runoff_coefficient'], data_dict['drainage']]],
                                  columns=['Water_Table', 'urbanization', 'Elevation',
                                           'precipitation', 'runoff_coefficient', 'drainage'])

        numerical_features = ['Elevation', 'precipitation', 'runoff_coefficient', 'drainage']
        input_data_scaled = input_data.copy()
        input_data_scaled[numerical_features] = scaler.transform(input_data[numerical_features])

        model_features = ['Water_Table', 'urbanization'] + numerical_features
        final_input = input_data_scaled[model_features]

        nb_prediction = rf_classifier.predict(final_input)
        nb_proba = rf_classifier.predict_proba(final_input)[0]

        waterlogging_probability = round(nb_proba[1] * 100, 2)

        logger.info(f"Prediction complete. Waterlogging probability: {waterlogging_probability}%")
        return {"waterlogging_probability": f"{waterlogging_probability}"}

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction error")
