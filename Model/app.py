import os

os.system("pip install --upgrade pip")
os.system("clear")
os.system("pip install tensorflow")
os.system("clear")
os.system("pip install keras")
os.system("clear")

import keras
import gradio as gr

model = keras.models.load_model("model.keras")


def recognize_digit(img):
    img = img.reshape(1, 784)
    img = img / 255
    prediction = model.predict(img).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}


gr.Interface(
    recognize_digit,
    inputs=gr.Sketchpad(),
    outputs=gr.Label(),
    title="Predict from 0 to 9",
).launch(share=True)
