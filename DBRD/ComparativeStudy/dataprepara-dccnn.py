import pandas as pd
import tensorflow as tf
import re
import string
#from nltk import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.python.keras.optimizers import TFOptimizer
import numpy as np
from nltk.corpus import stopwords
import string
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Conv3D
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling3D
from keras.layers import MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers import concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim
import csv
labeldup=[]

def clean_doc(input_text):
    # Tokenize the input document
    words = input_text.split()

    # Remove punctuation from words
    table = str.maketrans('', '', string.punctuation)
    words = [word.translate(table) for word in words]

    # Remove non-alphabetic words
    words = [word for word in words if word.isalpha()]

    # Define custom stop words
    custom_stop_words = ['java', 'com', 'org']

    # Add custom stop words to NLTK stop words
    stop_words = list(set(stopwords.words('english')))
    stop_words.extend(custom_stop_words)

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Stem words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Filter out short words
    words = [word for word in words if len(word) > 1]

    return words



reader1 = df['bug_1_text']
reader2 = df['bug_2_text']
rownumber = 0
train = []
matrix_b1 = []
matrix_b2 = []
traindata = []
train_1 = []
train_2 = []
len_token = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


reader1 = df['bug_1_text']
reader2 = df['bug_2_text']
rownumber = 0
traindata = []

for row1, row2 in zip(reader1, reader2):

    for c1 in row1:  # bug report 1
        cleanrow1 = clean_doc(c1)
    traindata.append(cleanrow1)

    for c2 in row2:  # bug report 2
        cleanrow2 = clean_doc(c2)
    traindata.append(cleanrow2)

vocab_size = 20
# traindata=np.array(traindata)
model = gensim.models.Word2Vec(traindata, size=vocab_size, min_count=1, sg=0)
# Cleaning
for row1, row2 in zip(reader1, reader2):

for c1 in row1:  # bug report 1
    cleanrow1 = clean_doc(c1)
for c2 in row2:  # bug report 2
    cleanrow2 = clean_doc(c2)

# Bug report 1

matrix_b1 = []

for i in cleanrow1:
    matrix1_b = model.wv[i]
    matrix_b1.append(matrix1_b)

train_1.append(matrix_b1)  # Bug report 1
rownumber = rownumber + 1
#       print("train_1",rownumber)

# Bug report 2

matrix_b2 = []

for i in cleanrow2:
    matrix2_nb = model.wv[i]
    matrix_b2.append(matrix2_nb)

train_2.append(matrix_b2)  # Bug report 2
rownumber = rownumber + 1
#    print("train_2",rownumber)
# f.close()

# t=np.array(train_2)
# t.shape
#padding
padded_test_1 = pad_sequences(test_1 , maxlen=max_length, padding='post')
#print(padded_test_1)
#padded_test_1.shape

padded_test_2 = pad_sequences(test_2 , maxlen=max_length, padding='post')
#print(padded_test_2)
#padded_test_2.shape
#padded_test_1.shape

#stack train_1 and train_2 (in pairs)
train=np.stack([padded_train_1, padded_train_2],axis=-1)
#train.shape
#stack test_1 and test_2
test=np.stack(([padded_test_1, padded_test_2]),axis=-1)
#test.shape