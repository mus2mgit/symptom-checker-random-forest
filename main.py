
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import difflib
import sys
import spacy

# Get the first command-line argument
try:
  arg = sys.argv[1]
except:
  arg = "I'm having headache with chest pain"



def get_closest_match(input_string: str, string_list: list) -> str:
  """
    Given an input string and a list of strings, find the string in the list
    that is the closest match to the input string based on a similarity threshold.
    If no string in the list is similar enough, return None.
    """
  closest_match = difflib.get_close_matches(input_string, string_list, n=1, cutoff=0.5)
  if closest_match:
    return closest_match[0]
  else:
    return None

# function to take column names as parameter and generate feature array
def mark_columns(standard_columns, specified_columns):
    """
    For each column name in the list, marks the cell in the row with a 1.
    For all other columns, marks the cell with a 0.
    Returns the modified row as a list.
    """
    marked_row = []
    for col in standard_columns:
        if get_closest_match(col,specified_columns) in specified_columns:
            marked_row.append(1)
        else:
            marked_row.append(0)
    return [marked_row]

def extract_symptoms(text,SYMPTOM_LIST):
  """
    Given a string of text and a list of symptom strings, extract the symptoms from the text by
    lemmatizing the tokens and checking if they are in the list of symptoms.
    Returns a list of matching symptoms.
    """
  doc = nlp(text)

  # Extract the symptoms by lemmatizing the tokens and checking if they are in the list of symptoms
  symptoms = [token.lemma_ for token in doc if get_closest_match(token.lemma_,SYMPTOM_LIST) in SYMPTOM_LIST]
  return symptoms


# Load and preprocess the data
# Load the training data from a CSV file
train_data = pd.read_csv('Training.csv')

# Split the training data into input features (X) and labels (y)
X_train = train_data.drop(['prognosis'], axis=1)
y_train = train_data['prognosis']

# Load the testing data from a CSV file
test_data = pd.read_csv('Testing.csv')

# Split the testing data into input features (X) and labels (y)
X_test = test_data.drop(['prognosis'], axis=1)
y_test = test_data['prognosis']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose a performance metric
metric = 'accuracy'

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
# Test the model
score = model.score(X_test, y_test)
print("Test score:", score)



# Extract the column names from the DataFrame
column_names = test_data.columns.tolist()[:-1]

# Load the English language model
nlp = spacy.load('en_core_web_md')


# Example usage
text = arg
symptoms = extract_symptoms(text,column_names)

data = mark_columns(column_names,symptoms)

predictions = model.predict(data)



# Print out the predictions
print("Predictions:", predictions)
