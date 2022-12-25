# Symptom Checker

This is a Python script that uses machine learning to predict the prognosis (illness or condition) based on a list of symptoms. It makes use of a Random Forest classifier trained on a dataset of symptoms and corresponding prognoses, and allows users to input a list of their symptoms in natural language and receive a prediction of their likely prognosis.

## **Dependencies**

The following libraries are required to run the script:

- tensorflow
- pandas
- scikit-learn
- difflib
- spacy

To install these dependencies, run the following command:

```
pip install tensorflow pandas scikit-learn difflib spacy
```

Additionally, the script uses the English language model from the **`en_core_web_md`** library of the **`spacy`** library. To install this model, run the following command:

```
python -m spacy download en_core_web_md
```

## **Running the script**

**Note: running the script trains the model everytime, if you want to use it you need to separate the model into another file, train it and save it to your disk. When you want to use that model you can then load it from a different file instead of training everytime.**

To run the script, open a terminal and navigate to the directory where the script is saved. Then, run the following command:

```
python main.py "list of symptoms"
```

Replace "list of symptoms" with a natural language list of your symptoms, separated by commas. For example:

```
python main.py "headache, chest pain, vomiting"

or

python main.py "I'm having a headache with chestpain."
```

The script will output a prediction of your likely prognosis.

## **Code structure**

The script is divided into three main sections:

1. Data preprocessing and model training
2. Symptom extraction
3. Prediction and output

In the first section, the script loads and preprocesses the training and testing data, trains a Random Forest classifier on the training data, and tests the model's performance on the testing data.

In the second section, the script defines the **`extract_symptoms`** function. This function takes in two arguments: a string of text and a list of symptom strings.

The function first uses the SpaCy library to tokenize the input text and create a document object. Then, it iterates over the tokens in the document, lemmatizes each token, and checks if the lemma is in the list of symptom strings. If a lemma is in the list of symptom strings, it is added to a list of matching symptoms. Finally, the function returns this list of matching symptoms.

In the third section, the script loads and preprocesses the data from two CSV files: **`Training.csv`** and **`Testing.csv`**. It splits the training data into input features (stored in the **`X_train`** variable) and labels (stored in the **`y_train`** variable). It also splits the testing data into input features (stored in the **`X_test`** variable) and labels (stored in the **`y_test`** variable).

The script then standardizes the input features using the **`StandardScaler`** class. It fits the scaler to the training data and then applies the transformation to both the training and testing data.

In the fourth section, the script defines a performance metric to use for evaluating the model's performance. In this case, the metric is **`accuracy`**.

In the fifth section, the script trains a **`RandomForestClassifier`** model on the standardized training data and labels. It then tests the model's performance on the standardized testing data and labels and prints out the score.

In the sixth section, the script extracts the column names from the testing data and stores them in the **`column_names`** variable. It then loads the English language model using **`spacy`**.

In the seventh section, the script takes a command-line argument and uses it as input text. It then extracts the symptoms from the text and stores them in the **`symptoms`** variable. It then generates a feature array based on the **`symptoms`** and the **`column_names`** using the **`mark_columns`** function.

Finally, in the eighth section, the script makes predictions on the generated feature array using the trained model and prints out the predictions.
