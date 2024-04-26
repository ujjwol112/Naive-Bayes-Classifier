import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('Tweets.csv')

# Extract relevant columns and drop missing values
df_dropped = df[['airline_sentiment', 'airline_sentiment_confidence', 'airline', 'text']]
df_dropped.dropna(inplace=True)

# Visualize sentiment distribution of airlines using bar plots
airline_counts = df_dropped['airline'].value_counts()
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Number of reviews for each Airlines in the dataset", fontweight='bold')
sns.countplot(df_dropped, x='airline')

# Visualize sentiment count of the airlines
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Sentiment Count of the Airlines", fontweight='bold')
senti_count = df_dropped['airline_sentiment'].value_counts()
sns.countplot(df_dropped, x='airline_sentiment')

# Categorize the confidence scores into equal frequency bins
num_bins_freq = 5
df_dropped['airline_sentiment_confidence_equal_freq'] = pd.qcut(df_dropped['airline_sentiment_confidence'], q=num_bins_freq, duplicates='drop')
bin_mapping = {bin_category: i for i, bin_category in enumerate(df_dropped['airline_sentiment_confidence_equal_freq'].unique())}
df_dropped['confidence_category'] = df_dropped['airline_sentiment_confidence_equal_freq'].map(bin_mapping)

# Convert airline_sentiment into numerical data using one-hot encoding
one_hot_encoded = pd.get_dummies(df_dropped['airline_sentiment'], prefix='sentiment').astype(int)
one_hot_encoded = pd.concat([df_dropped, one_hot_encoded], axis=1)

# Create a mapping dictionary to encode sentiment labels
sentiment_mapping = {
    'positive': 2,
    'neutral': 1,
    'negative': 0
}
one_hot_df = df_dropped.copy()
one_hot_df['airline_sentiment_encoded'] = df_dropped['airline_sentiment'].map(sentiment_mapping)

# Calculate and visualize sentiment rates for each airline
pivot_table_counts = one_hot_df.pivot_table(index='airline', columns='airline_sentiment', aggfunc='size', fill_value=0)
airline_counts = one_hot_df['airline'].value_counts()
sentiment_rates = pivot_table_counts.div(airline_counts, axis=0)
print(sentiment_rates)
sentiment_rates.plot.barh()
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Rate of each sentiment for the respective airlines", fontweight='bold')

# Visualize the relationship between sentiment and confidence scores
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Confidence score for the sentiment label", fontweight='bold')
plt.scatter(df_dropped['airline_sentiment'], df_dropped['airline_sentiment_confidence'])
plt.xlabel('airline_sentiment')
plt.ylabel('airline_sentiment_confidence')
plt.show()

# Visualize the distribution of airline_sentiment_confidence and its count
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Distribution of the airline_sentiment_confidence and its count", fontweight='bold')
sns.histplot(df_dropped, x='airline_sentiment_confidence', kde=True)

# Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function for text preprocessing: lowercase, remove non-alphabetic characters and punctuation, lemmatization
def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word.isalpha()])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Apply text preprocessing to the 'text' column
df_dropped['text'] = df_dropped['text'].apply(preprocess_text)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
new_text = vectorizer.fit_transform(df_dropped['text'])

# Create a new feature dataset with one-hot encoding for airlines
final_one_hot = pd.get_dummies(df_dropped['airline'], prefix='airline').astype(int)
final_one_hot = pd.concat([df_dropped.drop(['airline'], axis=1), final_one_hot], axis=1)

# Gaussian Naive Bayes Classifier
df_forGNB = final_one_hot.drop(['airline_sentiment', 'airline_sentiment_confidence_equal_freq', 'text', 'confidence_category'], axis=1)

classes = df_dropped['airline_sentiment']

# Perform train-test split for GNB
X_train, X_test, Y_train, Y_test = train_test_split(df_forGNB, classes, test_size=0.2, random_state=42)

# Initialize and train the Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train, Y_train)

# Visualize correlation matrix for GNB features
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Correlation Matrix for Gaussian NB", fontweight='bold')
corMat = df_forGNB.corr()
sns.heatmap(corMat, cmap='Blues', annot=True)
plt.show()

# Predict on the test data and calculate accuracy, classification report, and confusion matrix
Y_pred = GNB.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_report(Y_test, Y_pred, target_names=['negative', 'neutral', 'positive']))

cm = confusion_matrix(Y_test, Y_pred)

# Visualize the confusion matrix for GNB
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Confusion Matrix for Gaussian NB", fontweight='bold')
sns.heatmap(cm, cmap="Blues", annot=True, fmt='.2f', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel("Prediced")
plt.ylabel("Actual")
plt.show()

# Multinomial Naive Bayes Classifier
df_forMNB = final_one_hot.drop(['airline_sentiment', 'airline_sentiment_confidence_equal_freq', 'text', 'airline_sentiment_confidence'], axis=1)

# Perform train-test split for MNB
mX_train, mX_test, mY_train, mY_test = train_test_split(df_forMNB, classes, test_size=0.2, random_state=42)

# Initialize and train the Multinomial Naive Bayes model
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(mX_train, mY_train)

# Predict on the test data and calculate accuracy, classification report, and confusion matrix
mY_pred = naive_bayes_classifier.predict(mX_test)
accuracy = accuracy_score(mY_test, mY_pred)
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_report(mY_test, mY_pred, target_names=['negative', 'neutral', 'positive']))

cm = confusion_matrix(mY_test, mY_pred)

# Visualize the confusion matrix for MNB
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Confusion Matrix for Multinomial NB", fontweight='bold')
sns.heatmap(cm, cmap="Blues", annot=True, fmt='.2f', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel("Prediced")
plt.ylabel("Actual")
plt.show()

# Naive Bayes for Text and Sentiment
y = df_dropped['airline_sentiment']

# Perform train-test split for text-based sentiment analysis
tX_train, tX_test, tY_train, tY_test = train_test_split(new_text, y, test_size=0.2, random_state=42)

# Initialize and train the Multinomial Naive Bayes model for text-based sentiment analysis
tnaive_bayes_classifier = MultinomialNB()
tnaive_bayes_classifier.fit(tX_train, tY_train)

# Predict on the test data and calculate accuracy, classification report, and confusion matrix
tY_pred = tnaive_bayes_classifier.predict(tX_test)
accuracy = accuracy_score(tY_test, tY_pred)
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_report(tY_test, tY_pred, target_names=['negative', 'neutral', 'positive']))

cm = confusion_matrix(tY_test, tY_pred)

# Visualize the confusion matrix for text-based sentiment analysis
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Confusion Matrix for Multinomial NB only using Text feature", fontweight='bold')
sns.heatmap(cm, cmap="Blues", annot=True, fmt='.2f', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel("Prediced")
plt.ylabel("Actual")
plt.show()

# Hybrid Naive Bayes Classifier
hyb_df = final_one_hot.drop(['airline_sentiment', 'text', 'airline_sentiment_confidence_equal_freq'], axis=1)
target_class = final_one_hot['airline_sentiment']

# Perform train-test split for hybrid Naive Bayes
hX_train, hX_test, hY_train, hY_test = train_test_split(hyb_df, target_class, test_size=0.2, random_state=42)

# Drop continuous and discrete features for each classifier
hX_train_cont = hX_train.drop('confidence_category', axis=1)
hX_train_disc = hX_train.drop('airline_sentiment_confidence', axis=1)
hX_test_cont = hX_test.drop('confidence_category', axis=1)
hX_test_disc = hX_test.drop('airline_sentiment_confidence', axis=1)

# Initialize Gaussian and Multinomial Naive Bayes classifiers
gnb_cont = GaussianNB()
mnb_disc = MultinomialNB()

# Train each classifier with different feature sets
gnb_cont = gnb_cont.fit(hX_train_cont, hY_train)
mnb_disc = mnb_disc.fit(hX_train_disc, hY_train)

# Get probabilities from both classifiers and combine them to make predictions
gnb_probab = gnb_cont.predict_proba(hX_test_cont)
mnb_probab = mnb_disc.predict_proba(hX_test_disc)
hY_predict_probab = gnb_probab * mnb_probab
hY_predict = np.argmax(hY_predict_probab, axis=1)
