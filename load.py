import tensorflow as tf
from tensorflow.keras.models import load_model


# Load the model
loaded = tf.saved_model.load("saved_model/my_model")
scores, titles = loaded(["97"])

print(f"Recommendations: {titles[0, :3]}")