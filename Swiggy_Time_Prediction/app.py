from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
import uvicorn

# -----------------------------
# Dummy preprocessing & model
# -----------------------------
# If real preprocessor exists, replace these lines with joblib.load(...)
preprocessor = StandardScaler()
model = DummyRegressor(strategy="mean")  # always predicts the mean
model.fit([[0]], [0])  # dummy fit to initialize

# Build pipeline
model_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', model)
])

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Swiggy Delivery Time Prediction")

class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

# Home endpoint
@app.get("/")
def home():
    return {"message": "Your string here"}

# Predict endpoint
@app.post("/predict")
def do_predictions(data: Data):
    # Convert input to DataFrame
    pred_data = pd.DataFrame([data.dict()])
    
    # Dummy cleaning step (replace with real cleaning if available)
    cleaned_data = pred_data.copy()
    
    # Get prediction
    prediction = model_pipe.predict(cleaned_data)[0]
    
    return {"predicted_delivery_time": prediction}

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1/docs", port=8000)