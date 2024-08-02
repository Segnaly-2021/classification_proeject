import io
import cv2 
import uvicorn  # noqa: F401
import numpy as np
import tensorflow as tf
from ResBlock.residual import ResidualBlock # noqa: F401
from fastapi import FastAPI, UploadFile, File, HTTPException


app = FastAPI(title="Handwriten_digit_Identifier")


@app.get("/")
def homepage():
    return (
        "Welcome to the home page!!! To test the app, go to  http://localhost:8080/docs"
    )


@app.post("/predict/")
async def prediction(file: UploadFile = File(...)):
    '''
    Allows an HTTP Post request. It takes an image as input and outputs the
    corresponding digit on it along with the level of confidence.
    '''
    filename = file.filename
    fileExtention = filename.split(".")[-1] in ("jpeg", "jpg", "png")
    if not fileExtention:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    # Read the image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())

    # Start the stream from the beginning (position zero)
    image_stream.seek(0)

    # Convert the bytes stream into a numpy array
    bytes_to_nparray = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(bytes_to_nparray, cv2.IMREAD_GRAYSCALE)

    # Convert the image into a tensor, rescale it and then reshape it
    image_tensor =  tf.convert_to_tensor(image, dtype=tf.float32)
    image_0_1 = (image_tensor/255.0)
    input_image = tf.expand_dims((tf.expand_dims(image_0_1, axis=0)),axis=3)    

    # Load the model
    model = tf.saved_model.load("./app/models/")
    pred_func = model.signatures['serving_default'] 

    # Prediction
    y_pred = np.argmax(pred_func(input_image )['output_0'].numpy())
    conf_prob = np.max(pred_func(input_image )['output_0'].numpy()) * 100

    message = (
        f"The handwritten digit you provided is predicted to be {y_pred} "
        f"with a confidence of: {conf_prob:.6f} %"
    )

    return {"message": message}