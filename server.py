"""server module"""
from fastapi import FastAPI
import tensorflow as tf
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except:
        pass


loaded = tf.saved_model.load("saved_model/my_model")

app = FastAPI()

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
