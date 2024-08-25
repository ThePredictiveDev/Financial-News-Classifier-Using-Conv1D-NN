# Financial News Classifier Using Conv1D with a Trainable Embedding Layer

This project demonstrates how to build a financial news sentiment classifier using a Convolutional Neural Network (CNN) with a trainable embedding layer. The model classifies financial news into three categories: positive, neutral, and negative. The model achieved 98% accuracy

## Table of Contents

- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Creation, Compilation, and Training](#model-creation-compilation-and-training)
- [Model Evaluation](#model-evaluation)
- [Model Saving](#model-saving)
- [Fetching Real-Time News and Sentiment Analysis](#fetching-real-time-news-and-sentiment-analysis)
- [License](#license)

## Installation

To install the necessary packages, use the following `requirements.txt`:

```plaintext
tensorflow==2.6.0
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
newsapi-python==0.2.6
```
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

## Data Preprocessing
### Load Financial Phrase Bank Data
To train the model, we use the Financial Phrase Bank dataset. The dataset contains labeled financial news sentences that are categorized as positive, neutral, or negative.
```python
import os
import pandas as pd

def load_financial_phrase_bank(data_dir, encoding='utf-8'):
    sentences = []
    sentiments = []

    for filename in os.listdir(data_dir):
        if filename.startswith("Sentences_"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding=encoding) as file:
                    for line in file:
                        line = line.strip()
                        if line:
                            sentence, sentiment = line.rsplit('@', 1)
                            sentences.append(sentence.strip())
                            sentiments.append(sentiment.strip())
            except UnicodeDecodeError as e:
                print(f"Error decoding {filename} with encoding {encoding}: {e}")
                continue

    df = pd.DataFrame({
        'sentence': sentences,
        'sentiment': sentiments
    })

    return df

# Usage example
data_dir = "Your Download Directory/FinancialPhraseBank-v1.0"
encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

for enc in encodings:
    print(f"Trying encoding: {enc}")
    df = load_financial_phrase_bank(data_dir, encoding=enc)
    if not df.empty:
        print(f"Successfully loaded data with encoding: {enc}")
        break
else:
    print("Failed to load data with tried encodings.")
```

### Sentiment Label Mapping and Train-Test Split
Map sentiment labels to numerical values and split the data into training and testing sets.
```python
from sklearn.model_selection import train_test_split

# Mapping sentiment labels to numerical values
label_mapping = {"positive": 1, "neutral": 0, "negative": -1}
df['sentiment'] = df['sentiment'].map(label_mapping)

# Train-test split
RANDOM_SEED = 42
df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)

print(f"Training samples: {len(df_train)}, Testing samples: {len(df_test)}")
```
### Encoding and Vectorization
Encode sentiment labels and vectorize the input sentences using TensorFlow's TextVectorization layer.
```python
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Encode sentiment labels
label_encoder = LabelEncoder()
df_train['sentiment'] = label_encoder.fit_transform(df_train['sentiment'])
df_test['sentiment'] = label_encoder.transform(df_test['sentiment'])

# Define the TextVectorization layer
max_features = 20000  # Maximum vocabulary size
sequence_length = 128  # Maximum sequence length

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Adapt the vectorization layer on the training data
vectorize_layer.adapt(df_train['sentence'].values)

# Vectorize the sentences
train_inputs = vectorize_layer(df_train['sentence'].values)
test_inputs = vectorize_layer(df_test['sentence'].values)

train_labels = tf.convert_to_tensor(df_train['sentiment'].values)
test_labels = tf.convert_to_tensor(df_test['sentiment'].values)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))

# Shuffle, batch, and prefetch the datasets
batch_size = 32

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

```
# Model Creation, Compilation, and Training
Create a Conv1D model with a trainable embedding layer for sentiment classification.
```python
from tensorflow.keras import layers, Model

class SentimentClassifier(Model):
    def __init__(self, vocab_size, embedding_dim, n_classes):
        super(SentimentClassifier, self).__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.conv = layers.Conv1D(128, 5, activation='relu')
        self.global_pool = layers.GlobalMaxPooling1D()
        self.dropout = layers.Dropout(0.5)
        self.classifier = layers.Dense(n_classes, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        return self.classifier(x)

# Initialize the model
vocab_size = len(vectorize_layer.get_vocabulary())
embedding_dim = 128
n_classes = len(label_encoder.classes_)

classifier_model = SentimentClassifier(vocab_size, embedding_dim, n_classes)

# Compile the model
classifier_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

# Train the model
history = classifier_model.fit(
    train_dataset,
    epochs=20,
    validation_data=test_dataset
)

```

## Model Evaluation
Evaluate the trained model on the test dataset and generate a classification report.

```python
from sklearn.metrics import classification_report
import numpy as np

# Evaluate the model
loss, accuracy = classifier_model.evaluate(test_dataset)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predict labels for the test set
y_pred_probs = classifier_model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

# Get the true labels
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Convert integer class labels to strings
target_names = [str(cls) for cls in label_encoder.classes_]

# Print classification report
print(classification_report(y_true, y_pred, target_names=target_names))

```

## Model Saving
Save the trained model to a file for future use.

```python
classifier_model.save('sentiment_classifier_model.keras')

```

## Fetching Real-Time News and Sentiment Analysis
Use the trained model to classify real-time financial news headlines fetched using the NewsAPI.
```python
from newsapi import NewsApiClient

class NewsFetcher:
    def __init__(self, api_key):
        self.newsapi = NewsApiClient(api_key=api_key)

    def fetch_latest_news(self, query='stock market'):
        all_articles = self.newsapi.get_everything(q=query,
                                                   language='en',
                                                   sort_by='publishedAt',
                                                   page_size=5)
        headlines = [article['title'] for article in all_articles['articles']]
        return headlines

news_fetcher = NewsFetcher(api_key="your_api_key")
headlines = news_fetcher.fetch_latest_news(query='stock market')

class SentimentAnalysisTrader:
    def __init__(self, model, vectorize_layer):
        self.model = model
        self.vectorize_layer = vectorize_layer

    def predict_sentiment(self, headlines):
        inputs = self.vectorize_layer(headlines)
        probs = self.model.predict(inputs)
        sentiment_scores = np.argmax(probs, axis=1)
        return sentiment_scores

    def decide_trade_action(self, sentiment_score):
        if sentiment_score == 2:
            return "buy"
        elif sentiment_score == 0:
            return "sell"
        else:
            return "hold"

# Initialize the trader
trader = SentimentAnalysisTrader(model=classifier_model, vectorize_layer=vectorize_layer)

# Predict sentiment and decide trade action
for headline in headlines:
    sentiment_score = trader.predict_sentiment([headline])
    action = trader.decide_trade_action(sentiment_score[0])
    print(f"Headline: {headline}\nSentiment: {sentiment_score[0]} -> Action: {action}\n")

```

## License
This project is licensed under the MIT License
