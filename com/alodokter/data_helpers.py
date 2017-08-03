import os
import re
import string
import numpy as np
import itertools
from collections import Counter
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

NEWLINE     = '\n'
SKIP_FILES  = {'cmds'}

le, bin_enc = None, None

def __clean_str(string):
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

def __read_files(path):
    """
    read training files
    """
    print('read_files: '+ path)
    for root, dir_names, file_names in os.walk(path):
        for dir_name in dir_names:
            read_files(os.path.join(root, dir_name))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    lines = []
                    f = open(file_path, encoding='latin-1')
                    for line in f:
                        lines.append(line)
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content

def __build_data_frame(path, classification):
    """
    build training data, classification name using directory name
    """
    rows  = []
    index = []
    for file_name, text in __read_files(path):
        rows.append({'text': __clean_str(text.strip()), 'class': __one_hot_encoder(classification).tolist()[0]})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

# def __setup_one_hot_encoder_class(path):
#     print('setup_one_hot_encoder_class...')
#     classification = []
#     for root, dir_names, file_names in os.walk(path):
#         for dir_name in dir_names:
#             print('append dir_name: '+ dir_name)
#             classification.append(dir_name)
#
#     # integer encoded
#     global le, bin_enc
#     le = LabelEncoder()
#     int_enc = le.fit_transform(classification)
#     int_enc = int_enc.reshape(len(int_enc), 1) # reshape to avoid DeprecationWarning
#
#     # binary encode
#     bin_enc = OneHotEncoder(sparse=False)
#     bin_enc.fit(int_enc)

def setup_one_hot_encoder_class(path):
    print('setup_one_hot_encoder_class...')
    classification = []
    for root, dir_names, file_names in os.walk(path):
        for dir_name in dir_names:
            print('append dir_name: '+ dir_name)
            classification.append(dir_name)

    # integer encoded
    global le, bin_enc
    le = LabelEncoder()
    int_enc = le.fit_transform(classification)
    int_enc = int_enc.reshape(len(int_enc), 1) # reshape to avoid DeprecationWarning

    # binary encode
    bin_enc = OneHotEncoder(sparse=False)
    bin_enc.fit(int_enc)

def __one_hot_encoder(classname):
    int_enc = le.transform([classname])
    int_enc = int_enc.reshape(len(int_enc), 1) # reshape to avoid DeprecationWarning
    return bin_enc.transform(int_enc)

def __prepare_data(path):
    setup_one_hot_encoder_class(path)
    data = DataFrame({'text': [], 'class': []})
    for root, dir_names, file_names in os.walk(path):
        for dir_name in dir_names:
            data = data.append(__build_data_frame(os.path.join(root, dir_name), dir_name))

    return data.reindex(np.random.permutation(data.index))

def load_data_and_labels(path):
    data = __prepare_data(path)
    return data['text'].tolist(), np.array(data['class'].tolist())

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def get_class_name(index):
    return le.classes_[index]
