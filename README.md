
This repo hosts the data and code for the paper "Unsupervised Convolutional Filter Learning For COVID-19 Classification".

Note: This paper is [accepted](https://github.com/saichandrapandraju/Convolutional-Auto-Encoding-for-detecting-COVID/blob/main/Acceptance_Letter_RIA%2013011.pdf) on 13th Oct'21 and will be published soon.

In this project, we pre-trained a Convolutional Autoencoder with publicly available chest X-ray images and used this pre-trained model to classify COVID-19 chest X-ray images using Transfer Learning. 

- Dataset used for pre-training can be downloaded from [here](https://storage.googleapis.com/cae_covid_classification/pretrain.zip).
- Dataset used for COVID-19 classification can be downloaded from [here](https://storage.googleapis.com/cae_covid_classification/covid_normal_pneumonia.zip). This is a multiclass dataset with COVID, NORMAL and PNEUMONIA classes.
- [Here](https://cae-covid-classification.herokuapp.com/) is a quick demo of our model (It'll take a minute to initially load the app because the Dyno has to start).
## Reproducing our work:

We implemented our entire pipeline with interactive Jupyter notebooks and to reproduce our work, here are the sequential notebooks to run:

1. **pretrain_cae/pretrain_cae.ipynb** downloads the pretraining dataset and trains a Convolutional Autoencoder. The final model is saved as **cae.h5** and will be used for next step. If you want to skip this step, you can download **cae.h5** from [here](https://storage.googleapis.com/cae_covid_classification/cae.h5) and place it in `pretrain_cae` folder.
2. **covid_classification/cae_covid_classification.ipynb** downloads the COVID-19 classification dataset and uses the encoder part of previous step for classification. The final classification model is saved as **covid_classification.h5**. This model can be downloaded from [here](https://storage.googleapis.com/cae_covid_classification/covid_classification.h5) if you want to skip the finetuning part and just want inference.
3. We used **deploy/*** files to deploy this classification model to [Heroku](https://www.heroku.com/). A quick inference of our model can be made with this demo app - https://cae-covid-classification.herokuapp.com/ (It'll take a minute to initially load the app because the Dyno has to start).


For complete details of approach, kindly refer our paper.
