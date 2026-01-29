import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("dog_breed_classifier_model.h5")

# IMPORTANT: class order comes from training folders
CLASS_NAMES = sorted([
    d for d in range(model.output_shape[-1])
])

IMG_SIZE = (224, 224)

def predict(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image) / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    top_indices = preds.argsort()[-3:][::-1]

    return {
        f"Breed {i}": float(preds[i])
        for i in top_indices
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Dog Breed Classification (MobileNetV2)",
    description="Upload a dog image to predict its breed. Model trained on the Oxford-IIIT Pet Dataset."
)

if __name__ == "__main__":
    demo.launch()
