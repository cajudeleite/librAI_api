import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
#import python-multipart
import cv2
import io
import os

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def translate_output(y_pred):

    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    max_index = np.argmax(y_pred,axis=1)
    translated_output = [labels[idx] for idx in max_index]
    return translated_output

@app.post("/predict")

async def predict(
        img: UploadFile=File(...)
    ):

    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)

    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    cv2.imwrite('api/img_test/test/img.jpg', cv2_img)

    """Function that makes the prediction of LibrAI and feeds the API"""

    ## Load the model

    model = load_model("api/model_v1")

    im_shape = (64, 64)
    seed = 10

    IMG_DIR = 'api/img_test'

    img_test = ImageDataGenerator(rescale=1/255)
    img_test = img_test.flow_from_directory(IMG_DIR, target_size=im_shape, shuffle=False, seed=seed)

    test = model.predict(img_test)

    translated_result = translate_output(test)

    os.remove('api/img_test/test/img.jpg')

    return {"prediction": translated_result}

    #return dict(dummy_prediction=float(y_pred))

@app.get("/")
def root():
    return dict(greeting="Welcome to LibrAI API!!")
