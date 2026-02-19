# **Periodontal Disease Prediction**

This project uses a **Convolutional Neural Network (CNN)** to detect periodontal (gum) disease from panoramic dental images. It performs **binary classification** (disease or no disease) and uses **5-Fold Cross-Validation** to properly evaluate model performance.

## **Project Files**

**model.py** – Trains the CNN model using K-Fold Cross-Validation and saves the best model.

**predict.py** – Loads the trained model and predicts the class for a new dental image.

**app.py** – Flask web application that allows users to upload an image and receive a prediction.

## **Features**

* Images resized to **128x128** and normalized
* **CNN-based** binary classification
* **5-Fold Cross-Validation**
* Evaluation using **Accuracy, Precision, and Recall**
* Training **accuracy and loss visualization**

## **Dataset**

**dataset/**

* **penyakit-periodontal/**
* **penyakit-non-periodontal/**

## **Usage**

Install required libraries: **tensorflow, numpy, matplotlib, scikit-learn, pandas, flask**.

Run **model.py** to train the model.
Run **predict.py** to test a new image.
Run **app.py** to start the web application.

## Author
Shashank H K
