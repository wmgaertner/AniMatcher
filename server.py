"""server module"""
from fastapi import FastAPI
import tensorflow as tf
from tensorflow.keras.models import load_model


app = FastAPI()

loaded = tf.saved_model.load("saved_model/my_model")

@app.get("/")
async def root():
    """root endpoint"""
    return {"message": "Hello World during the coronavirus pandemic!"}

@app.get("/{user_id}")
async def get_recommendations(user_id: str):
    """get recommendations endpoint"""
    scores, titles = loaded([user_id])
    titles = titles.numpy().tolist()
    return {"titles": titles[0]}
