# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
df = pd.read_csv(r"C:\Users\Harshal\Desktop\TYCS Files\Harshal_TYCS_Files\IR\IR_Prac5\Test.csv")

# Combine 'covid' and 'fever' columns into a single feature column
data = df["covid"] + " " + df["fever"]
X = data.astype(str)  # Test data
y = df['flu']  # Labels

# Splitting the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting data into bag-of-words format
vectorizer = CountVectorizer()  # Initialize CountVectorizer

# Transform training and test data
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Initialize and train the Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Load another dataset to test the model
data1 = pd.read_csv(r"C:\Users\Harshal\Desktop\TYCS Files\Harshal_TYCS_Files\IR\IR_Prac5\Test.csv")
new_data = data1["covid"] + " " + data1["fever"]

# Transform new data using the trained vectorizer
new_data_counts = vectorizer.transform(new_data.astype(str))

# Make predictions on the new dataset
predictions = classifier.predict(new_data_counts)

# Output the predictions
print("Predictions for new data:")
print(predictions)

# Evaluate the model using test data
accuracy = accuracy_score(y_test, classifier.predict(X_test_counts))
print(f"\nAccuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, classifier.predict(X_test_counts)))

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions, columns=['flu_prediction'])

# Concatenate predictions with the original test DataFrame
data1 = pd.concat([data1, predictions_df], axis=1)

# Save the updated DataFrame to a new CSV file
data1.to_csv(r"C:\Users\Harshal\Desktop\TYCS Files\Harshal_TYCS_Files\IR\IR_Prac5\Test1.csv", index=False)
