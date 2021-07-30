import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

labels = ['COVID', 'NORMAL', 'PNEUMONIA']

if 'covid_classification.h5' not in os.listdir('.'):
    os.system('wget https://storage.googleapis.com/cae_covid_classification/covid_classification.h5')
if 'images' not in os.listdir('.'):
    os.system('mkdir images')

model = tf.keras.models.load_model('covid_classification.h5')

def image_classifier(img):
  im = Image.fromarray(img)
  im.save('images/recent.jpeg')
  img = tf.keras.preprocessing.image.load_img('images/recent.jpeg', target_size=(224,224))
  img_array = tf.keras.preprocessing.image.img_to_array(img).astype('float32')/255
  img_array = tf.expand_dims(img_array, 0)
  prediction = model.predict(img_array).flatten()
  return {labels[i]: float(prediction[i]) for i in range(len(labels))}

iface = gr.Interface(
    image_classifier, 
    gr.inputs.Image(), 
    gr.outputs.Label(num_top_classes=len(labels)),
    capture_session=True,
    interpretation="default",
    title='COVID-19 classification using Convolutional Autoencoding Approach',
    description='This is a demo app for COVID-19 classification'
    )

if __name__ == "__main__":
    iface.launch()