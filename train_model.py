# Objective
#- Aim is to train a model on the data labelled with weak supervision

import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

## Load manually labelled cluster label

df_labels = pd.read_csv("data/manually_labelled_clusters.csv")

cluster_label_desc_map = {}
for key, value in zip(list(df_labels['cluster_number']), list(df_labels['cluster_label'])):
    cluster_label_desc_map[key] = value


 ## Load clustered sentences
df_entities = pd.read_csv("data/all_clustered_entities.csv")
df_entities['label'] = df_entities['cluster_label'].apply(lambda x: cluster_label_desc_map[x])
#print(df_entities.head())   
## Train test split
column = 'processed_sentences'
df_entities = df_entities[~df_entities[column].isnull()].reset_index(drop=True)
df_entities.label.value_counts()

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_entities, stratify=list(df_entities['label']), test_size=0.3, random_state=42)

df_train.label.value_counts()

df_test.label.value_counts()
## Extract TFIDF features

from sklearn.feature_extraction.text import TfidfVectorizer

# Extract TFIDF features with 1-4 grams
TFIDF_PARAMS = {
    'strip_accents': 'ascii',
    'stop_words': 'english',
    'sublinear_tf': True,
    'ngram_range': (1, 3),
    'min_df': 0.0005
}
vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
print(vectorizer)

tfidf_model = vectorizer.fit(df_train[column])
print(tfidf_model)

vocab_length = len(tfidf_model.vocabulary_)
print ("Length of vocabulary: {}".format(vocab_length))

with open('trained_models/tfidf_classifier_extractor.pkl', 'wb') as filepath:
    pickle.dump(tfidf_model, filepath, protocol=4)

X_train = tfidf_model.transform(df_train[column])
y_train = list(df_train.label)

X_test = tfidf_model.transform(df_test[column])
y_test = list(df_test.label)

## Experiment with Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)

with open('trained_models/random_forest_model.pkl', 'wb') as filepath:
    pickle.dump(clf, filepath, protocol=4)

scores = clf.predict_proba(X_test)
y_true = pd.DataFrame(scores,columns=clf.classes_).idxmax(axis=1)
print (classification_report(y_test, y_true, digits=4))

## Experiment with Support Vector Machine

from sklearn.svm import SVC
clf = SVC(probability=True)
clf = clf.fit(X_train, y_train)

with open('trained_models/svm_model.pkl', 'wb') as filepath:
    pickle.dump(clf, filepath, protocol=4)

scores = clf.predict_proba(X_test)
y_true = pd.DataFrame(scores,columns=clf.classes_).idxmax(axis=1)
print (classification_report(y_test, y_true, digits=4))

## Experiment with XGBoost
from xgboost import XGBClassifier
clf = XGBClassifier(n_jobs= -1, objective='multi:softmax')
clf = clf.fit(X_train, y_train)

with open('trained_models/xgboost_model.pkl', 'wb') as filepath:
    pickle.dump(clf, filepath, protocol=4)

scores = clf.predict_proba(X_test)
y_true = pd.DataFrame(scores,columns=clf.classes_).idxmax(axis=1)
print (classification_report(y_test, y_true, digits=4))

## Experiment with Bidirectional LSTM

# One hot encoding
y_train_nn = pd.get_dummies(y_train)
ordered_classes = list(y_train_nn.columns)

y_train_nn.head()

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras import initializers, regularizers, constraints, optimizers, layers

inp = Input(shape=(vocab_length,))
x = Embedding(vocab_length, 100)(inp)
x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(len(ordered_classes), activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

check_pointer = ModelCheckpoint(filepath='trained_models/bilstm_model.hdf5', verbose=1, save_best_only=True)
training_args = {
    'validation_split': 0.2,
    'batch_size': 32,
    'epochs': 1,
    'callbacks': [check_pointer],
}

model.fit(X_train.toarray(), y_train_nn.values, **training_args)

scores = model.predict(X_test.toarray())
y_true = pd.DataFrame(scores, columns=ordered_classes).idxmax(axis=1)
print (classification_report(y_test, y_true, digits=4))