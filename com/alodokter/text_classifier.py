import os
import re
import pickle
import numpy
import string
import nltk
from collections import OrderedDict
from sklearn.externals import joblib
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

import pymongo
from pymongo import MongoClient

class TextClassifier:

    NEWLINE = '\n'
    WHITESPACE = ' '
    SKIP_FILES = {'cmds'}
    CORPUS_PATH  = 'corpus/interest/'

    '" loading stopwords data "'
    def __init__(self):
        self.data = None
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vocabs = {}

    def __clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    '" read training files "'
    def __read_files(self, path):
        print('processing path: '+ path)
        for root, dir_names, file_names in os.walk(path):
            for dir_name in dir_names:
                self.__read_files(os.path.join(root, dir_name))
            for file_name in file_names:
                if file_name not in TextClassifier.SKIP_FILES:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        lines = []
                        f = open(file_path, encoding='latin-1')
                        for line in f:
                            lines.append(line)
                        f.close()
                        content = TextClassifier.NEWLINE.join(lines)
                        yield file_path, content

    '" build training data, classification name using directory name "'
    def __build_data_frame(self, path, classification):
        rows  = []
        index = []
        self.vocabs[classification] = []
        for file_name, text in self.__read_files(path):
            rows.append({'text': self.__clean_str(text), 'class': classification})
            index.append(file_name)
            self.vocabs[classification] += text.split(' ')

        data_frame = DataFrame(rows, index=index)
        return data_frame

    def conditional_freq_dist(self):
        return nltk.ConditionalFreqDist((interest, word) for interest in self.vocabs.keys() for word in self.vocabs[interest])

    def save(self):
        joblib.dump(self.data, 'pickled/_interest_data.pkl')
        joblib.dump(self.pipeline, 'pickled/_interest_pipeline.pkl')

    def load(self):
        self.data = joblib.load('pickled/_interest_data.pkl')
        self.pipeline = joblib.load('pickled/_interest_pipeline.pkl')

    def prepare_data(self, path=CORPUS_PATH):
        data = DataFrame({'text': [], 'class': []})
        for root, dir_names, file_names in os.walk(path):
            for dir_name in dir_names:
                data = data.append(self.__build_data_frame(os.path.join(root, dir_name), dir_name))

        self.data = data.reindex(numpy.random.permutation(data.index))

    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['text'].values, self.data['class'].values, test_size=0.4, random_state=5)

        # self.pipeline = Pipeline([('count_vectorizer',  CountVectorizer(ngram_range=(1,  2))), ('tfidf_transformer',  TfidfTransformer()), ('classifier',  MultinomialNB())])
        self.pipeline = Pipeline([('count_vectorizer',  CountVectorizer()), ('classifier',  MultinomialNB())])
        # self.pipeline = Pipeline([('vectorizer',  CountVectorizer()), ('classifier',  SVC(kernel='rbf'))])
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self, text, n_samples=1, confidence_level=0):
        if n_samples == 1:
            return self.pipeline.predict([text])
        else:
            probs = self.pipeline.predict_proba([text])
            predictions = {}
            for prob in probs:
                for idx, val in enumerate(prob):
                    predictions[self.pipeline.steps[1][1].classes_[idx]] = val

            if confidence_level == 0:
                return sorted(predictions.items(), key=lambda kv: kv[1], reverse=True)[:n_samples]
            else:
                results = []
                for item in sorted(predictions.items(), key=lambda kv: kv[1], reverse=True)[:n_samples]:
                    if item[1] >= confidence_level:
                        results.append((item[0], item[1]))
                return results

    def score(self):
        print('Score: %0.2f' % self.pipeline.score(self.X_test, self.y_test))

    def f1_score(self):
        k_fold = KFold(n=len(self.data), n_folds=6)
        scores = []
        for train_indices, test_indices in k_fold:
            train_text = self.data.iloc[train_indices]['text'].values
            train_y = self.data.iloc[train_indices]['class'].values

            test_text = self.data.iloc[test_indices]['text'].values
            test_y = self.data.iloc[test_indices]['class'].values

            self.pipeline = Pipeline([('vectorizer',  CountVectorizer()), ('classifier',  MultinomialNB())])
            self.pipeline.fit(train_text, train_y)
            predictions = self.pipeline.predict(test_text)

            score = f1_score(test_y, predictions, average=None)
            scores.append(score)
        print('f1 Score: %0.2f' % sum(scores)/len(scores))

    def tagging(self):
        client = MongoClient('[server ip]', 27017)
        db = client.alomobile
        client.alomobile.authenticate('[username]', '[password]', mechanism='SCRAM-SHA-1')

        # questions = db.questions.find().limit( 100 )
        questions = db.questions.find({ '_type': 'Core::Question' })
        #questions = db.questions.find({ '_type': 'Core::Question', 'interest': { '$exists': False } })
        for question in questions:
            text = question['title'] +' '+ question['content']
            print(text)
            prediction_interest = self.predict(text, 5, 0.5)
            print(prediction_interest)
            if len(prediction_interest) > 0:
                print('tagging '+ str(question['_id']) +' with '+ prediction_interest[0][0])
                db.questions.update( { '_id': question['_id'] }, {"$set": { 'interest': prediction_interest[0][0] }}, upsert=False )
                print('')
