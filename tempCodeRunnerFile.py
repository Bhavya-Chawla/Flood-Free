from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the trained models
try:
    nb_classifier = joblib.load('sid.pkl')  # Load Naive Bayes model
    label_encoders = {  # Load label encoders for categorical features
        'Water_Table': joblib.load('Water_Table_label_encoder.pkl'),
        'urbanization': joblib.load('urbanization_label_encoder.pkl')
    }
    scaler = joblib.load('scaler.pkl')  # Load scaler
    logger.info("Model, label encoders, and scaler loaded successfully.")
except FileNotFoundError:
    logger.error("One or more required files are missing.")
    raise HTTPException(status_code=500, detail="Required files missing.")
except Exception as e:
    logger.error(f"Error loading files: {e}")
    raise HTTPException(status_code=500, detail="Error loading files.")

# Define the input schema with root validators for validation
class WeatherData(BaseModel):
    drainage: int
    Water_Table: str  # This should be a string
    urbanization: str  # This should be a string
    precipitation: int
    runoff_coefficient: float
    Elevation: int


    @model_validator(mode='before')
    def check_negative_values(cls, values):
        # Ensure no negative values for numeric fields
        for field, value in values.items():
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{field} must be non-negative.")
        return values

    @model_validator(mode='before')
    def check_empty_fields(cls, values):
        # Ensure Water_Table and urbanization are not empty
        if not values.get('Water_Table') or not values.get('urbanization'):
            raise ValueError("Water_Table and urbanization cannot be empty.")
        return values

# POST endpoint for prediction
@app.post("/predict")
def predict(data: WeatherData):
    logger.info(f"Received input data: {data.dict()}")
    try:
        # Transform categorical fields with label encoders
        data_dict = data.dict()
        data_dict['Water_Table'] = label_encoders['Water_Table'].transform([data_dict['Water_Table']])[0]
        data_dict['urbanization'] = label_encoders['urbanization'].transform([data_dict['urbanization']])[0]

        # Construct a DataFrame from the input
        input_data = pd.DataFrame([[data_dict['drainage'], data_dict['Water_Table'], data_dict['urbanization'], 
                                    data_dict['precipitation'], data_dict['runoff_coefficient'], data_dict['Elevation']]], 
                                  columns=['drainage', 'Water_Table', 'urbanization', 'precipitation', 
                                           'runoff_coefficient', 'Elevation'])

        # Scale numerical features using the pre-trained scaler
        input_data_scaled = scaler.transform(input_data)

        # Predict using Naive Bayes
        nb_prediction = nb_classifier.predict(input_data_scaled)
        nb_proba = nb_classifier.predict_proba(input_data_scaled)[0]  # Get class probabilities

        # Extract the probability of class 1 (waterlogging)
        waterlogging_probability = round(nb_proba[1] * 100, 2)

        logger.info(f"Naive Bayes waterlogging class (class 1) probability: {waterlogging_probability}%")
        return {"waterlogging_probability": f"{waterlogging_probability}%"}

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction error")
