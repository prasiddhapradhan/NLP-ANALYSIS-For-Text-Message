# NLP-ANALYSIS-For-Text-Message

Project Overview

This project involves analyzing text message (SMS) data using Natural Language Processing (NLP) techniques to gain insights into the dataset. The dataset clean_nus_sms.csv contains SMS messages along with metadata such as message length, country of origin, and timestamp.

Using Python and various NLP libraries, we perform preprocessing and analysis to extract meaningful insights, including:

Sentiment Analysis

Tokenization

Named Entity Recognition (NER)

Part-of-Speech (POS) Tagging

Topic Modeling

Word Vectorization

Installation & Setup

This project was developed and run on VS Code (Visual Studio Code) on macOS using a virtual environment (venv).

1. Set Up Virtual Environment

To ensure a clean working environment, create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate  # For macOS/Linux

2. Install Dependencies

Run the following command to install necessary Python libraries:

pip install pandas numpy nltk scikit-learn matplotlib seaborn spacy

If nltk is missing data resources, install them manually:

python -m nltk.downloader stopwords
python -m nltk.downloader punkt
python -m nltk.downloader wordnet

If you encounter SSL certificate errors while downloading NLTK data on macOS, run:

/Applications/Python\ 3.11/Install\ Certificates.command

(Replace 3.11 with your Python version if different.)

Project Workflow

Step 1: Load the Dataset

The dataset is loaded using pandas:

import pandas as pd

# Load the dataset
file_path = "clean_nus_sms.csv"
df = pd.read_csv(file_path)
print(df.info())
print(df.head())

Step 2: Preprocessing the Text Data

Convert text to lowercase

Remove special characters & stopwords

Tokenization

Lemmatization

Example:

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

df["cleaned_message"] = df["Message"].apply(preprocess_text)

Step 3: Sentiment Analysis

Using TextBlob to determine the sentiment polarity of messages:

from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment'] = df['cleaned_message'].apply(get_sentiment)

Step 4: Named Entity Recognition (NER) with SpaCy

import spacy
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

df['entities'] = df['cleaned_message'].apply(extract_entities)

Step 5: Visualization

Using matplotlib and seaborn to plot insights:

import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment distribution
plt.figure(figsize=(8,5))
sns.histplot(df['sentiment'], bins=20, kde=True)
plt.title("Sentiment Score Distribution")
plt.show()

Step 6: Word Cloud Generation

To visualize frequently used words:

from wordcloud import WordCloud
text = " ".join(df["cleaned_message"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

Troubleshooting

1. ModuleNotFoundError (Pandas, NLTK, etc.)

If a library is missing, install it using:

pip install <module_name>

2. SSL Certificate Error While Downloading NLTK Data

Run:

/Applications/Python\ 3.11/Install\ Certificates.command

3. Spacy Model Not Found

Download the required SpaCy model:

python -m spacy download en_core_web_sm

Conclusion

This project demonstrated the power of Natural Language Processing (NLP) by analyzing and extracting insights from SMS data. Techniques such as sentiment analysis, word vectorization, topic modeling, and named entity recognition helped uncover patterns in the dataset. Future improvements could include machine learning-based text classification and automated topic clustering.

🔹 Author: Prasiddha Pradhan🔹 Technologies Used: Python, NLTK, Pandas, Matplotlib, SpaCy, Scikit-learn🔹 Date: February 2025
