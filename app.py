import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request

# Load the data
raw_mail_data = pd.read_csv('email.csv')

# Preprocess the data
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Convert category to numeric
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Filter out non-integer values in the 'Category' column
mail_data = mail_data[mail_data['Category'].apply(lambda x: str(x).isdigit())]

# Split data into features and target
x = mail_data['Message']
y = mail_data['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Vectorize the text data
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# Convert target variables to integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Train the logistic regression model
model = LogisticRegression()
model.fit(x_train_features, y_train)

# Prediction function
def predict_email_category(email_message):
    features = feature_extraction.transform([email_message])
    prediction = model.predict(features)[0]
    return prediction

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_message = request.form['email_message']
    prediction = predict_email_category(email_message)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


