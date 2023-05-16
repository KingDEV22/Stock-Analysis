import threading
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
import numpy as np
import time
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D, Bidirectional, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO (or any desired level)
    # Set the log message format
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Set the date and time format
)

logging.info("Dataset loading")
df = pd.read_csv('stock_data.csv')
df['Text'] = df['Text'].astype(str)
df['Sentiment'] = df['Sentiment'].astype(np.int64)

daily_data = pd.read_csv('./raw_news/dailynews.csv')
daily_data['Text'] = daily_data['Text'].astype(str)
logging.info("Dataset merging")
data = pd.concat([df, daily_data], ignore_index=True)
data.drop_duplicates(subset="Text",
                     keep="first", inplace=True)

logging.info("Dataset loaded and merge")


def text_data_cleaning(sentence):
    lemmatizer = WordNetLemmatizer()
    sent = preprocess_text(sentence)
    doc = nltk.word_tokenize(sent)
    lemma = [lemmatizer.lemmatize(word, pos="v") for word in doc]
    return append_message(lemma)


def preprocess_text(sen):
    to_remove = ['up', 'down', 'low', 'high', 'below', 'less', 'fall']
    new_stopwords = set(stopwords.words('english')).difference(to_remove)
    sentence = str(sen).lower()
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub('rs|cr|crore|point|points|pt|stock', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    pattern = re.compile(r'\b(' + r'|'.join(new_stopwords) + r')\b\s*')
    sentence = pattern.sub('', sentence)
    return sentence


def append_message(text):
    str = " "
    return (str.join(text))


def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def vectorize_data(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)
    tf_idf_vect = TfidfVectorizer()
    X_train = tf_idf_vect.fit_transform(X_train)
    X_test = tf_idf_vect.transform(X_test)
    pickle.dump(tf_idf_vect, open("vectorizer.pkl", "wb"))
    logging.info("Vectorization completed")
    return X_train, X_test, y_train, y_test


def create_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = vectorize_data(X, y)
    clf2 = LogisticRegression(
        C=10, multi_class='multinomial', penalty='l2', solver='lbfgs')
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    logging.info("LR is trained")
    print(accuracy_score(y_pred, y_test))


def create_SDG(X, y):
    X_train, X_test, y_train, y_test = vectorize_data(X, y)
    clf1 = SGDClassifier(alpha=0.0001, loss='modified_huber')
    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    logging.info("SDG is trained")
    print(accuracy_score(y_pred, y_test))


def padding_sequence(X):
    vocab = set()
    for x in X:
        vocab.add(x)
    tokenizer = Tokenizer(num_words=len(vocab))
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    max_length = max(len(sequence) for sequence in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
    pickle.dump(sequences, open("sequences.pkl", "wb"))
    pickle.dump(max_length, open("max_length.pkl", "wb"))
    logging.info("Padded sequences completed")
    return vocab, padded_sequences, max_length


def create_LSTM(vocab, max_length, X_train, X_test, y_train, y_test):
    # Define model architecture

    model = Sequential()
    model.add(Embedding(len(vocab), 150, input_length=max_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    optimizer = Adam(
        learning_rate=2e-03,  # HF recommendation
        epsilon=1e-08,
        clipnorm=1.0
    )
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
    logging.info("lstm started")
    # Train model
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=5, batch_size=3000, verbose=2)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')
    y_pred = model.predict(X_test)
    label_scores = {}
    for i, label in enumerate(set(y)):
        idx = np.where(y_test == i)[0]
        label_scores[label] = np.mean(y_pred[idx, i])

    # print the label specific score
    for label in label_scores:
        print('Score for LSTM label {}: {:.2f}%'.format(
            label, label_scores[label] * 100))

    logging.info("lstm trained")


def create_BiGRU(vocab, max_length, X_train, X_test, y_train, y_test):

    # Define model architecture
    model = Sequential()
    model.add(Embedding(len(vocab), 128, input_length=max_length))
    model.add(SpatialDropout1D(0.5))
    model.add(Bidirectional(GRU(128, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dense(len(set(y)), activation='softmax'))
    optimizer = Adam(
        learning_rate=1e-03,  # HF recommendation
        epsilon=1e-08,
        clipnorm=1.0
    )
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
    logging.info("bigru started")
    # Train model
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=10, batch_size=3000, verbose=2)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')
    y_pred = model.predict(X_test)
    label_scores = {}
    for i, label in enumerate(set(y)):
        idx = np.where(y_test == i)[0]
        label_scores[label] = np.mean(y_pred[idx, i])

    # print the label specific score
    for label in label_scores:
        print('Score for BiGRU label {}: {:.2f}%'.format(
            label, label_scores[label] * 100))
    logging.info("bigru trained")


def create_BiGRU1(vocab, max_length, X_train, X_test, y_train, y_test):

    # Define model architecture
    model = Sequential()
    model.add(Embedding(len(vocab), 128, input_length=max_length))
    model.add(SpatialDropout1D(0.5))
    model.add(Bidirectional(GRU(128, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(y)), activation='softmax'))
    optimizer = Adam(
        learning_rate=2e-03,  # HF recommendation
        epsilon=1e-08,
        clipnorm=1.0
    )
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
    logging.info("bigru1 started")
    # Train model
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=8, batch_size=3000, verbose=2)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')
    y_pred = model.predict(X_test)
    label_scores = {}
    for i, label in enumerate(set(y)):
        idx = np.where(y_test == i)[0]
        label_scores[label] = np.mean(y_pred[idx, i])

    # print the label specific score
    for label in label_scores:
        print('Score for BiGRU1 label {}: {:.2f}%'.format(
            label, label_scores[label] * 100))
    logging.info("bigru1 trained")


if __name__ == "__main__":
    data['Tok_text'] = data['Text'].apply(text_data_cleaning)
    data['Tok_text'] = data['Tok_text'].astype(str)
    X = data['Tok_text']
    y = data['Sentiment']
    vocab, padded_sequence, max_length = padding_sequence(X)
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequence, y, test_size=0.3, random_state=42, stratify=y)

    create_logistic_regression(X, y)
    create_SDG(X, y)
    create_LSTM(vocab, max_length, X_train, X_test, y_train, y_test)
    create_BiGRU(vocab, max_length, X_train, X_test, y_train, y_test)
    create_BiGRU1(vocab, max_length, X_train, X_test, y_train, y_test)
