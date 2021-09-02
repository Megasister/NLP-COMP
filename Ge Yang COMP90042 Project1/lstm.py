# This code is sourced from
# https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
# Modified and customized by Ge Yang in input part,
# struture part and output part
# to deal with task requirement.
# Detail implementations and tunings are all done individually.
# all parts from online is referenced in annotation.

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import keras.backend as K

# Others
from nltk.stem.snowball import SnowballStemmer
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import os
import json
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


stopwords = set(stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
unigram_vectorizer = DictVectorizer()


def get_tweet_text_from_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data


notebook_path = os.path.abspath("modelling.ipynb")

train_path = os.path.join(os.path.dirname(notebook_path), "train.json")
etl_train_path1 = os.path.join(os.path.dirname(notebook_path), "etl-input1.json")
# etl_train_path2 = os.path.join(os.path.dirname(notebook_path), "etl-input2.json")
etl_train_path3 = os.path.join(os.path.dirname(notebook_path), "small-etl-input.json")
# etl_train_path4 = os.path.join(os.path.dirname(notebook_path), "big-etl-input.json")
dev_path = os.path.join(os.path.dirname(notebook_path), "dev.json")
test_path = os.path.join(os.path.dirname(notebook_path), "test-unlabelled.json")

train_data = get_tweet_text_from_json(train_path)
etl1_train_data = get_tweet_text_from_json(etl_train_path1)
# etl2_train_data = get_tweet_text_from_json(etl_train_path2)
etl3_train_data = get_tweet_text_from_json(etl_train_path3)
# etl4_train_data = get_tweet_text_from_json(etl_train_path4)
dev_data = get_tweet_text_from_json(dev_path)
test_data = get_tweet_text_from_json(test_path)


def filter_abnml(text):
    num = '[0-9!"#$%&\'()*+,-./:;<=>?，。?*【】《》？""''！[\\]^_`{|}~]+'
    stript = re.sub('\n', ' ', text)
    stript = re.sub(num, '', stript)
    return stript


def paragraph_to_sentence(events):

    sentences_bag = []
    labels = []
    for _, event in events.items():
        text = event['text']
        try:
            label = event['label']
        except:
            label = -1
        sentences = nltk.sent_tokenize(text)
        sentences_bag.append(sentences)
        labels.append(label)

    return sentences_bag, labels


def paragraph_to_bgr_bag(events):

    bgr_bag = []
    labels = []
    for _, event in events.items():
        text = event['text']
        try:
            label = event['label']
        except:
            label = -1
        sentences = nltk.sent_tokenize(text)
        bgr_bag.append(' '.join(sentences))
        labels.append(label)

    return bgr_bag, labels


for k, v in etl1_train_data.items():
    if v['text'] != '':
        train_data[k+'e1'] = v

# for k, v in etl2_train_data.items():
#     if v['text'] != '':
#         train_data[k+'e2'] = v

for k, v in etl3_train_data.items():
    if v['text'] != '':
        train_data[k+'e3'] = v
#
# for k, v in etl4_train_data.items():
#     if v['text'] != '':
#         train_data[k+'e4'] = v

# source from online
def clean_text(text):
    ## Remove puncuation
    text = text.translate(string.punctuation)

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    text = [w for w in text if not w in stopwords and len(w) >= 3]

    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text


train_bags, train_labels = paragraph_to_bgr_bag(train_data)
dev_bags, dev_labels = paragraph_to_bgr_bag(dev_data)
test_bags, _ = paragraph_to_bgr_bag(test_data)

print(len(train_labels))

# apply the above function to df['text']
train_bags = list(map(lambda x: clean_text(x), train_bags))
dev_bags = list(map(lambda x: clean_text(x), dev_bags))
print(train_bags)


# Create sequence
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(train_bags)

train_sequences = tokenizer.texts_to_sequences(train_bags)
dev_sequences = tokenizer.texts_to_sequences(dev_bags)
test_sequences = tokenizer.texts_to_sequences(test_bags)
train_pad = pad_sequences(train_sequences, maxlen=50)
dev_pad = pad_sequences(dev_sequences, maxlen=50)
test_pad = pad_sequences(test_sequences, maxlen=50)


# source from online
def get_f1(y_true, y_pred):
    # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# Network architecture
# source from online, modified by Ge Yang
model = Sequential()
model.add(Embedding(20000, 100, input_length=50))
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[get_f1])
# Fit the model
model.fit(train_pad, np.array(train_labels),
          validation_data=(dev_pad, np.array(dev_labels)),
          epochs=3)

# predict test set with trained model
# predict = model.predict_classes(test_pad)
# output_test = {}
# zipped = zip(test_data.items(), predict)
# for (k, _), pred in zipped:
#     # print(k)
#     # output_test['dev-'+str(k[4:])] = {"label": int(pred[0])}
#     # print(output_test)
#     output_test[k] = {"label": int(pred[0])}
#
# with open('test-output.json', 'w') as outfile:
#     json.dump(output_test, outfile)
#
# print('file write finished')

# word_embds = model.layers[0].get_weights()
# word_list = []
# for word, i in tokenizer.word_index.items():
#     word_list.append(word)
#
# X_embedded = TSNE(n_components=2).fit_transform(word_embds)
# number_of_words = 1000
# trace = go.Scatter(
#     x = X_embedded[0:number_of_words,0],
#     y = X_embedded[0:number_of_words, 1],
#     mode = 'markers',
#     text= word_list[0:number_of_words]
# )
# layout = dict(title = 't-SNE 1 vs t-SNE 2 for sirst 1000 words ',
#               yaxis = dict(title='t-SNE 2'),
#               xaxis = dict(title='t-SNE 1'),
#               hovermode= 'closest')
# fig = dict(data = [trace], layout= layout)
# py.iplot(fig)
