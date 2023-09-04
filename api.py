import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#test http://127.0.0.1:8000/predict?dummy_feature1=2&dummy_feature2=6


#app.state.model = load_model()

@app.get("/predict")

def predict(
        dummy_feature1: float,
        dummy_feature2: float,
    ):

    """Function that makes the prediction of LibrAI and feeds the API"""

    #Dummy API - to test

    X = pd.DataFrame(locals(), index=[0], columns = ["dummy_feature1","dummy_feature2"])

    y_pred = X["dummy_feature1"] + X["dummy_feature2"]


    # Complete after model is done
    
    ## DataFrame creating and preprocessing

    pass #X_pred = pd.DataFrame(locals(), index=[0])
    pass #model = app.state.model

    pass #X_processed = preprocess_features(X_pred)
    pass #y_pred = model.predict(X_processed)


    return dict(dummy_prediction=float(y_pred))

@app.get("/")
def root():
    return dict(greeting="Welcome to LibrAI API!!")