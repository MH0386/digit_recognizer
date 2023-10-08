import os
import keras
import gradio as gr

os.system("pip install tensorflow")

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
).launch()
