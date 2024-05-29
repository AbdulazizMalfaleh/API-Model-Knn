from fastapi import FastAPI, HTTPException

app = FastAPI()

# GET request for root path
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

# GET request for items path
@app.post("/items/")
def create_item(item: dict):
    return {"item": item}

import joblib 
model = joblib.load('knn_model.joblib') 
scaler = joblib.load('Models/scaler.joblib')

from pydantic import BaseModel

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    Year: int
    Engine_Size: float
    Mileage: float
    Type: str
    Make: str
    Options: str

def preprocessing(input_features: InputFeatures, scaler):
    dict_f = {
        'Year': input_features.Year,
        'Engine_Size': input_features.Engine_Size,
        'Mileage': input_features.Mileage,
        'Type_Accent': input_features.Type == 'Accent',
        'Type_Land_Cruiser': input_features.Type == 'Land Cruiser',
        'Make_Hyundai': input_features.Make == 'Hyundai',
        'Make_Mercedes': input_features.Make == 'Mercedes',
        'Options_Full': input_features.Options == 'Full',
        'Options_Standard': input_features.Options == 'Standard'
    }
    
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Scale the input features
    scaled_features = scaler.transform([features_list])
    
    return scaled_features

from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/predict")
def predict(input_features: InputFeatures):
    try:
        scaled_features = preprocessing(input_features, scaler)
        return {"scaled_features": scaled_features.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict") 
def predict(input_features: InputFeatures): 
    return preprocessing(input_features)

from pydantic import BaseModel
import joblib
import os
from fastapi import FastAPI, HTTPException

# Load the scaler (assuming it is already trained and saved)
scaler = joblib.load('Models/scaler.joblib')

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    Year: int
    Engine_Size: float
    Mileage: float
    Type: str
    Make: str
    Options: str

# Define the preprocessing function
def preprocessing(input_features: InputFeatures):
    dict_f = {
        'Year': input_features.Year,
        'Engine_Size': input_features.Engine_Size,
        'Mileage': input_features.Mileage,
        'Type_Accent': input_features.Type == 'Accent',
        'Type_Land_Cruiser': input_features.Type == 'Land Cruiser',
        'Make_Hyundai': input_features.Make == 'Hyundai',
        'Make_Mercedes': input_features.Make == 'Mercedes',
        'Options_Full': input_features.Options == 'Full',
        'Options_Standard': input_features.Options == 'Standard'
    }
    
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Scale the input features
    scaled_features = scaler.transform([features_list])
    
    return scaled_features

app = FastAPI()

@app.post("/predict")
def predict(input_features: InputFeatures):
    try:
        scaled_features = preprocessing(input_features)
        return {"scaled_features": scaled_features.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# Load the scaler and model (assuming they are already trained and saved)
scaler = joblib.load('Models/scaler.joblib')
model = joblib.load('Models/knn_model.joblib')

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    Year: int
    Engine_Size: float
    Mileage: float
    Type: str
    Make: str
    Options: str

# Define the preprocessing function
def preprocessing(input_features: InputFeatures):
    dict_f = {
        'Year': input_features.Year,
        'Engine_Size': input_features.Engine_Size,
        'Mileage': input_features.Mileage,
        'Type_Accent': input_features.Type == 'Accent',
        'Type_Land_Cruiser': input_features.Type == 'Land Cruiser',
        'Make_Hyundai': input_features.Make == 'Hyundai',
        'Make_Mercedes': input_features.Make == 'Mercedes',
        'Options_Full': input_features.Options == 'Full',
        'Options_Standard': input_features.Options == 'Standard'
    }
    
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Scale the input features
    scaled_features = scaler.transform([features_list])
    
    return scaled_features

app = FastAPI()

@app.post("/predict")
async def predict(input_features: InputFeatures):
    try:
        # Preprocess the input features
        data = preprocessing(input_features)
        
        # Predict using the loaded model
        y_pred = model.predict(data)
        
        # Return the prediction
        return {"pred": y_pred.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)


 