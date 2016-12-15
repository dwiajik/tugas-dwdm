import random
import re
import time

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC

english_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class Classifier:
    def clean_text(self, text):
        regex = re.compile('\shttp.+\s')
        text = regex.sub(' ', text)
        regex = re.compile('[^a-zA-Z\']')
        text = regex.sub(' ', text)
        return text

    def get_features(self, text):
        text = self.clean_text(text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(token) \
            for token in tokens \
            if '\'s' not in token \
            and '\'t' not in token]
        features = {}
        for token in tokens:
            if token not in english_stopwords and len(token) > 1:
                features["{}".format(token)] = tokens.count(token)
        return (features)

    def __init__(self, labeled_texts):
        random.shuffle(labeled_texts)
        train_set = [(self.get_features(text), category) for (text, category) in labeled_texts]
        self.data_count = len(train_set)

        start_time = time.time()
        self.svm_classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
        self.training_time = round(time.time() - start_time, 2)

    def classify(self, text):
        return self.svm_classifier.classify(self.get_features(text))

    def get_training_time(self):
        return self.training_time

    def get_data_count(self):
        return self.data_count
