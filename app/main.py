
# Imports
import io
import cv2
import uvicorn
import cvlib as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from residual import ResidualBlock 
from fastapi import FastAPI, UploadFile, File, HTTPException


app= FastAPI(title='Handwriten_digit_Identifier')

@app.get("/")
def homepage():
    return "Welcome to the home page!!! To test the app, go to  http://localhost:8080/docs"

@app.post("/predict/")
async def prediction(file: UploadFile = File(...)):

    filename = file.filename
    fileExtention = filename.split('.')[-1] in ('jpeg','jpg','png')
    if not fileExtention:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    # Read the image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())
    
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    image_0_1 = (image/255).reshape(-1,28,28,1)

    model = keras.models.load_model("./app/model.keras")
    
    # Prediction 
    y_pred = np.argmax(model.predict(image_0_1))
    conf_prob = np.max(model.predict(image_0_1))*100

    message= f"The handwriten digit you provided is predicted to be {y_pred} with a confidence of:{conf_prob:.6f} %"
    
    return {'message': message}
