import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
#import python-multipart
import cv2
import io

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

@app.post("/predict")

async def predict(
        img: UploadFile=File(...)
    ):

    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)

    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    cv2_img = cv2_img/255

    desired_shape = (32, 32, 3)

    resized_img = cv2.resize(cv2_img, dsize = (desired_shape[1], desired_shape[0]))

    resized_img = np.expand_dims(resized_img,axis=0)

    """Function that makes the prediction of LibrAI and feeds the API"""

    # Load the model

    model = load_model("model_v1")

    y_pred = model.predict(resized_img)

    labels = ['airplane',
          'automobile',
          'bird',
          'cat',
          'deer',
          'dog',
          'frog high on DMT',
          'horse',
          'ship',
          'truck']

    max_index = np.argmax(y_pred,axis=1)

    translated_result = labels[max_index[0]]

    print("-------------------------------------------")
    print(max_index)

    return {"prediction": translated_result}

    #return dict(dummy_prediction=float(y_pred))

@app.get("/")
def root():
    return dict(greeting="Welcome to LibrAI API!!")
