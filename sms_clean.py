import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("clean_nus_sms.csv")

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "id"])

# Convert 'length' column to integer
df["length"] = pd.to_numeric(df["length"], errors="coerce")

# Handle missing values
df = df.dropna(subset=["Message"])  # Drop rows where 'Message' is NaN

# Convert date format
df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m")

# Display dataset summary
print(df.info())
print(df.head())

#Exploratory Data Analysis (EDA)#
#message length distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["length"], bins=50, kde=True)
plt.xlabel("Message Length")
plt.ylabel("Count")
plt.title("Distribution of SMS Message Lenghts")
plt.show()

#word cloud for most frequent words
text = " ".join(msg for msg in df["Message"].dropna())

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in SMS Messages")
plt.show()

#Text Preprocessing#

nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        tokens = word_tokenize(text)  # Tokenization
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        return " ".join(tokens)
    return ""

df["cleaned_message"] = df["Message"].apply(preprocess_text)

print(df[["Message", "cleaned_message"]].head())

#Sentiment Analysis#

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["cleaned_message"].apply(get_sentiment)

# Sentiment distribution
sns.countplot(data=df, x="sentiment", palette=["red", "blue", "green"])
plt.title("Sentiment Analysis of SMS Messages")
plt.show()

#Named entity recognition (NER)#
nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

df["named_entities"] = df["cleaned_message"].apply(extract_named_entities)
print(df[["Message", "named_entities"]].head())


#Topic modeling with latent dirichlet allocation (LDA)#
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X = vectorizer.fit_transform(df["cleaned_message"])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Display top words in each topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx}:")
    print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))
    
    
#Checking Similarity Between Messages#
# Compute similarity matrix
similarity_matrix = cosine_similarity(X)

# Find most similar messages
def find_similar_messages(index, top_n=3):
    similar_indices = similarity_matrix[index].argsort()[-top_n-1:-1][::-1]
    return df.iloc[similar_indices][["Message"]]

sample_index = 10  # Example message
print(f"Original Message: {df.iloc[sample_index]['Message']}")
print("\nMost Similar Messages:")
print(find_similar_messages(sample_index))
                                

    




