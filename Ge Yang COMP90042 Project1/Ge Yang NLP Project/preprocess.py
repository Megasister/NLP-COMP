# Regular expression in function filter_abnml
# is sourced from online.
# other module is original

# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import keras
# from sklearn import model_selection
# from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import wordnet
import nltk.tokenize as tk
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
bigram_vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
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
    # source from online
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
        sentences = nltk.sent_tokenize(filter_abnml(text))
        bgr_bag.append(' '.join(sentences))
        labels.append(label)

    return bgr_bag, labels


# mix dev data in training
# dev_test = {}
# for k, v in dev_data.items():
#     if int(k[4:]) > 0:
#         train_data[k] = v
#     else:
#         dev_test[k] = v
# dev_data = dev_test


for k, v in etl1_train_data.items():
    if v['text'] != '':
        train_data[k+'e1'] = v

# for k, v in etl2_train_data.items():
#     if v['text'] != '':
#         train_data[k+'e2'] = v
#
for k, v in etl3_train_data.items():
    if v['text'] != '':
        train_data[k+'e3'] = v

# for k, v in etl4_train_data.items():
#     if v['text'] != '':
#         train_data[k+'e4'] = v


train_bags, train_labels = paragraph_to_bgr_bag(train_data)
dev_bags, dev_labels = paragraph_to_bgr_bag(dev_data)
test_bags, _ = paragraph_to_bgr_bag(test_data)

print(len(train_labels))


def get_wordnet_pos(word):
    # from https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    # rewrite by Ge Yang
    tag = nltk.tag.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    word_list = []
    for word in words:
        if word.lower() not in stopwords:
            tag = get_wordnet_pos(word)
            lemma = lemmatizer.lemmatize(word, tag)
            word_list.append(lemma)

    return " ".join(word_list)


def bag_of_words(bags):
    event_list = []
    for bag in bags:
        event_map = defaultdict(int)
        for sentence in bag:
            preprocessed_words = tk.word_tokenize(sentence)
            preprocessed_words = [word for word in preprocessed_words if word not in stopwords]
            for word in preprocessed_words:
                event_map[word] += 1

        event_list.append(event_map)
    return event_list
#
#
# train_event_list = bag_of_words(train_bags)
# dev_event_list = bag_of_words(dev_bags)
# test_event_list = bag_of_words(test_bags)


train_vec = bigram_vectorizer.fit_transform(train_bags)
dev_vec = bigram_vectorizer.transform(dev_bags)
test_vec = bigram_vectorizer.transform(test_bags)

'''
    C = np.arange(1, 0, -0.1)
for c in C:
    print('C = ', c)
'''

# kernels = ['linear', 'rbf', 'poly']
# for kernel in kernels:
# clf = SVC(C=10)

clf = LogisticRegression()

model = clf.fit(train_vec, train_labels)
accuracy = accuracy_score(model.predict(dev_vec), dev_labels)
report = classification_report(model.predict(dev_vec), dev_labels)
print(report)

predict = model.predict(dev_vec)

for (_, v), pre, label in zip(dev_data.items(), predict, dev_labels):
    if pre != label:
        print(pre, label)

for (_, v), pre, label in zip(dev_data.items(), predict, dev_labels):
    if pre != label:
        print(pre, label)
        print(v['text'])


# predict test set with trained model
# predict = model.predict(test_vec)
# output_test = {}
# zipped = zip(test_data.items(), predict)
# for (k, _), pred in zipped:
#     # output_test['dev-'+str(k[5:])] = {"label": int(pred)}
#     output_test[k] = {"label": int(pred)}
#
# with open('test-output.json', 'w') as outfile:
#     json.dump(output_test, outfile)
#
# print('file write finished')










