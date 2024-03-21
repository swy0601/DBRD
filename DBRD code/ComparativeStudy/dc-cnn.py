import tensorflow as tf
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
tf.keras.metrics.Precision( thresholds=None, top_k=None, class_id=None, name=None, dtype=None)
tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)
tf.keras.metrics.FalseNegatives( thresholds=None, name=None, dtype=None)
tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)

import tensorflow as tf
import tensorflow
from keras import layers
from keras import Input
from tensorflow import keras
from keras.layers import Dense
from tensorflow.keras.layers import Reshape
# define the model

import keras.backend as k
import tensorflow as tf

#define CNN model
def define_model():
	# channel 1
    #inputs1 = Input(shape=(tt.shape))
    inputs1 = Input(shape=(300,20,2))
    #inputs2 = Input(shape=(300,20))
    #inputs12=concatenate([inputs1,inputs2])
    #inputs12=tf.stack(([inputs1,inputs2]),axis=-1)
    #inputs12=tf.keras.backend.stack(([inputs1,inputs2]),axis=-1)
    #(300,20,2) tf.keras.backend.stack(     x,     axis=0 )
    #print(inputs1.shape)
    print("inputs1: ",inputs1.shape)
    conv1 = Conv2D(filters=100, kernel_size=(1,20), padding='valid', activation='relu', data_format='channels_last')(inputs1)
    #print(conv1)# "(300,1,100)"
    print("conv1:",conv1.shape)
    x= layers.Reshape((300,100,1))(conv1)
    print("x:",x.shape)
    conv11 = Conv2D(filters=200, kernel_size=(1,100), padding='valid', activation='relu')(x)
    print("conv11:",conv11.shape)
    pool11 = MaxPooling2D(pool_size=(300, 1))(conv11)
    print("pool11:",pool11.shape)
    flat11 = Flatten()(pool11)
    print(flat11.shape)
    # Branch12
    conv12 = Conv2D(filters=200, kernel_size=(2,100), padding='valid', activation='relu')(x)
    print("conv12:",conv12.shape)
    pool12 = MaxPooling2D(pool_size=(299, 1))(conv12)
    print("pool12:",pool12.shape)
    flat12 = Flatten()(pool12)
    # Branch13
    conv13 = Conv2D(filters=200, kernel_size=(3,100), padding='valid', activation='relu')(x)
    print("conv13:",conv13.shape)
    pool13 = MaxPooling2D(pool_size=(298, 1))(conv13)
    print("pool13:", pool13.shape)
    flat13 = Flatten()(pool13)
    # Branch21
    conv2 = Conv2D(filters=100, kernel_size=(2,20), padding='valid', activation='relu', data_format='channels_last')(inputs1)
    print("conv2: ", conv2.shape )
    x2 = layers.Reshape((299,100,1))(conv2)
    print("x2:", x2.shape)
    conv21 = Conv2D(filters=200, kernel_size=(1,100), padding='valid', activation='relu')(x2)
    print("conv21 :",conv21.shape)
    pool21 = MaxPooling2D(pool_size=(299, 1))(conv21)#o2-maxpooling
    print("pool21: ",pool21.shape)
    flat21 = Flatten()(pool21)
    #Branch22
    conv22 = Conv2D(filters=200, kernel_size=(2,100), padding='valid', activation='relu')(x2)
    print("conv22:", conv22.shape)
    pool22 = MaxPooling2D(pool_size=(298, 1))(conv22)
    print("pool22:",pool22.shape)
    flat22 = Flatten()(pool22)
    #Branch23
    conv23 = Conv2D(filters=200, kernel_size=(3,100), padding='valid', activation='relu')(x2)
    print("conv23:",conv23.shape)
    pool23 = MaxPooling2D(pool_size=(297, 1))(conv23)
    print("pool23: ",pool23.shape)
    flat23 = Flatten()(pool23)
    #Branch31
    conv3 = Conv2D(filters=100, kernel_size=(3,20), padding='valid', activation='relu', data_format='channels_last')(inputs1)
    print("conv3",conv3.shape)
    x3 = layers.Reshape((298,100,1))(conv3)
    print("x3",x3.shape)
    conv31 = Conv2D(filters=200, kernel_size=(1,100), padding='valid', activation='relu')(x3)
    print("conv31: ",conv31.shape)
    pool31 = MaxPooling2D(pool_size=(298, 1))(conv31)
    print("pool31",pool31.shape)
    flat31 = Flatten()(pool31)
    #Branch32
    conv32 = Conv2D(filters=200, kernel_size=(2,100), padding='valid', activation='relu')(x3)
    print("conv32: ",conv32.shape)
    pool32 = MaxPooling2D(pool_size=(297, 1))(conv32)
    print("pool32",pool32.shape)
    flat32 = Flatten()(pool32)
    #Branch33
    conv33 = Conv2D(filters=200, kernel_size=(3,100), padding='valid', activation='relu')(x3)
    print("conv33: ",conv33.shape)
    pool33 = MaxPooling2D(pool_size=(296, 1))(conv33)
    print("pool33",pool33.shape)
    flat33 = Flatten()(pool33)
    # merge 'May be merge pools then flatten
    merged = concatenate([flat11, flat12, flat13, flat21, flat22, flat23, flat31, flat32, flat33])
    print("merged: ",merged.shape)
    # interpretation
    dense1 = Dense(300, activation='relu')(merged)
    dense2 = Dense(100, activation='relu')(dense1)
    outputs = Dense(1, activation='sigmoid')(dense2)
    print(outputs.shape)
    #model = Model(inputs=(inputs1, inputs2), outputs=outputs) #think about this
    model = Model(inputs=inputs1, outputs=outputs)
    # compile
    model.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='binary_crossentropy' , metrics=[tf.keras.metrics.Accuracy(),tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TruePositives() ])

    # summarize
    print(model.summary())
    return model

model=define_model()
history = model.fit(train, trainLabels, batch_size=64, epochs=1, shuffle=True)  ### Sh## Shuffle =true

end = time.time()

print(f"Runtime of the program is {end - start}")

from sklearn import metrics

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

print("Testing: ")

# evaluate testing set
testing_score = model.evaluate(test, testLabels)
print('testing evaluation:', testing_score)

predictions = model.predict(test)
predictions
predictions = (predictions > 0.5)
np.set_printoptions(threshold=np.inf)

y_pred = [int(val) for val in predictions]
# if y_pred == True:
#       y_pred= 1
# else:
#       y_pred= 0

# print(y_pred)

print(y_pred)

# accuracy
acc = accuracy_score(testLabels, predictions)
print("accuracy %.2f", (acc * 100))

# confusion matrix
cm = confusion_matrix(testLabels, predictions)
print(cm)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
# all_sample_title = 'Accuracy Score: {0}'.format(score)
# plt.title(all_sample_title, size = 15);

# classification report
report = classification_report(testLabels, predictions)
print("cm", cm)
print("report", report)

# from sklearn.model_selection import cross_val_predict, cross_val_score
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Define your model (assuming you have already created it as 'model')
#
# # Perform k-fold cross-validation
# k = 5  # Choose the number of folds (e.g., 5-fold cross-validation)
# cv_scores = cross_val_score(model, test, testLabels, cv=k)
#
# # Print the cross-validation scores
# print("Cross-validation scores:", cv_scores)
# print("Mean accuracy: {:.2f}%".format(100 * cv_scores.mean()))
#
# # Get predictions for each fold
# y_pred = cross_val_predict(model, test, testLabels, cv=k)
#
# # accuracy
# acc = accuracy_score(testLabels, y_pred)
# print("Accuracy: {:.2f}%".format(100 * acc))
#
# # confusion matrix
# cm = confusion_matrix(testLabels, y_pred)
# print("Confusion Matrix:")
# print(cm)
# plt.figure(figsize=(9, 9))
# sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
#
# # classification report
# report = classification_report(testLabels, y_pred)
# print("Classification Report:")
# print(report)
