# **Symptom Checker**

This repository contains a machine learning model for identifying common symptoms from natural language descriptions. The model is trained on a dataset of symptom descriptions and their corresponding labels, and can be used to predict the likely symptoms for a given input text.

## **Requirements**

- Python 3.6 or higher
- TensorFlow 2.0 or higher
- spaCy 2.3 or higher

## **Installation**

To install the required libraries, run the following command:

```
pip install -r requirements.txt

```

## **Usage**

To use the model, run the **`predict_symptoms.py`** script with the input text as a command-line argument:

```
python predict_symptoms.py "I have a headache and a fever"

```

The script will output a list of predicted symptoms, separated by commas.

## **Training**

To train the model on your own dataset, you will need to prepare the data in the following format:

```
description,label
"I have a headache and a fever",fever,headache
"I have a rash on my skin",rash

```

The **`description`** column should contain the natural language text, and the **`label`** column should contain a comma-separated list of corresponding symptom labels.

To train the model, run the **`train.py`** script with the path to the data file as a command-line argument:

```
python train.py data.csv

```

The script will save the trained model to a file called **`symptom_checker.h5`**.

## **Evaluation**

To evaluate the model's performance on a test dataset, you can use the **`evaluate.py`** script. This script expects the test data to be in the same format as the training data, and will output the model's accuracy and F1 score on the test set.

```
python evaluate.py test_data.csv

```
