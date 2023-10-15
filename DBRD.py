
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from transformers import TFAutoModel
from tqdm import tqdm
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout,Reshape,Softmax,LSTM, Dropout, Bidirectional
from keras.models import Model
import numpy as np
BERT = TFAutoModel.from_pretrained('bert-base-uncased')
for layer in BERT.layers:
    layer.trainable = False #使BERT层的参数不参与反相传播
def embedding(inputs_id):
    if tf.test.is_gpu_available():
        device = "/gpu:0"
    else:
        device = "/cpu:0"
    with tf.device(device):
      outputs = BERT(inputs_id, output_hidden_states=True, training=False)
      hidden = outputs[2]
      embeddings = tf.stack(hidden, axis=0)
      final = tf.reduce_sum(embeddings[-4:], axis=0)
    del hidden,embeddings,outputs
    return final


def Siamese():
    input_shape = (200,)
    input_dtype = tf.int32
    input = tf.keras.layers.Input(shape=input_shape, dtype=input_dtype)
    X_1 = embedding(input)
    X_1_1 = Conv1D(200, kernel_size=1, strides=1, activation='relu')(X_1)
    X_1_1 = tf.keras.layers.BatchNormalization()(X_1_1)
    X_1_1 = MaxPooling1D(pool_size=300,padding='same')(X_1_1)
    X_1_1 = Flatten()(X_1_1)

    X_1_2 = Conv1D(200, kernel_size=2, strides=1, activation='relu')(X_1)
    X_1_2 = tf.keras.layers.BatchNormalization()(X_1_2)
    X_1_2 = MaxPooling1D(pool_size=299,padding='same')(X_1_2)
    X_1_2 = Flatten()(X_1_2)

    X_1_3 = Conv1D(200, kernel_size=3, strides=1, activation='relu')(X_1)
    X_1_3 = tf.keras.layers.BatchNormalization()(X_1_3)
    X_1_3 = MaxPooling1D(pool_size=298,padding='same')(X_1_3)
    X_1_3 = Flatten()(X_1_3)

    X_1 = tf.keras.layers.Concatenate(axis=-1)([X_1_1, X_1_2])
    output = tf.keras.layers.Concatenate(axis=-1)([X_1, X_1_3])
    # output = lstm_layer(X_1)
    output = Dropout(0.6)(output)
    output = Dense(300, activation='relu')(output)
    output = tf.keras.layers.BatchNormalization(axis=-1)(output)

    output = Dropout(0.4)(output)
    output = Dense(100, activation='relu')(output)
    output = tf.keras.layers.BatchNormalization(axis=-1)(output)


    sub_model = Model(input, output)

    # Then define the tell-digits-apart model
    left = Input(shape=input_shape)
    right = Input(shape=input_shape)

    # The vision model will be shared, weights and all
    left_output = sub_model(left)
    right_output = sub_model(right)

    out =  tf.keras.layers.Concatenate(axis=-1)([left_output,right_output])
    prediction = tf.keras.layers.Dense(2, activation='softmax')(out)

    siamese_model = Model([left, right], prediction)
    return siamese_model

DBRD = Siamese()
DBRD.summary()


def train_step(inputs, labels, model, optimizer, loss_fn, p1,p2,p3):#再加入一个验证集
  if tf.test.is_gpu_available():
    device = "/gpu:0"
  else:
    device = "/cpu:0"
  with tf.device(device):
    with tf.GradientTape() as tape:
        left, right = inputs
        pred = model([left, right])
        loss = loss_fn(labels, pred)
  if labels[0] == 1:
    loss += tf.dtypes.cast(tf.math.reduce_mean(p1), tf.float32)
    loss += tf.dtypes.cast(tf.math.reduce_mean(p2), tf.float32)
    loss += tf.dtypes.cast(tf.math.reduce_mean(p3), tf.float32)

    # Compute gradients and update weights
    # Original label 1 corresponds to the unique hot encoding vector [1, 0], indicating duplication
    # The original label 0 corresponds to the unique encoding vector [0,1], indicating no repetition

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Update metrics
    accuracy_metric.update_state(labels, pred)
    loss_tracker.update_state(loss)



# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)




# Train the model
epochs = 100
batch_size1 = 64
# batch_size2 = 128
# batch_size3 = 256

patience = 10
wait = 0
best = 0
best_loss = 100
# best_auc = 10000

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

TP = tf.keras.metrics.TruePositives()
TN = tf.keras.metrics.TrueNegatives()
FP = tf.keras.metrics.FalsePositives()
FN = tf.keras.metrics.FalseNegatives()
def evaluate(left, right, labels, model, loss_fn, p1, p2, p3, batch_size):

    for i in tqdm(range(0, len(left), batch_size)):
        val_left_batch = left[i:i+batch_size]
        val_right_batch = right[i:i+batch_size]
        val_labels_batch = labels[i:i+batch_size]
        val_p1_batch = p1[i:i+batch_size]
        val_p2_batch = p2[i:i+batch_size]
        val_p3_batch = p3[i:i+batch_size]
        val_pred = model([val_left_batch, val_right_batch])
        val_loss = loss_fn(val_labels_batch,val_pred)+ tf.dtypes.cast(tf.math.reduce_mean(val_p1_batch),tf.float32)+ tf.dtypes.cast(tf.math.reduce_mean(val_p2_batch),tf.float32) + tf.dtypes.cast(tf.math.reduce_mean(val_p3_batch),tf.float32)
        val_accuracy_metric.update_state(val_labels_batch, val_pred)

        TP.update_state(val_labels_batch[:, 0], val_pred[:, 0])
        TN.update_state(val_labels_batch[:, 0], val_pred[:, 0])
        FP.update_state(val_labels_batch[:, 0], val_pred[:, 0])
        FN.update_state(val_labels_batch[:, 0], val_pred[:, 0])
        val_loss_tracker.update_state(val_loss)
    print("TP:",TP.result().numpy())
    print("TN:",TN.result().numpy())
    print("FP:",FP.result().numpy())
    print("FN:",FN.result().numpy())
    TP.reset_states()
    TN.reset_states()
    FP.reset_states()
    FN.reset_states()

#额外训练参数
# w1 = tf.Variable(tf.random.normal([1], mean=0.0, stddev=1.0))
# w2 = tf.Variable(tf.random.normal([1], mean=0.0, stddev=1.0))
# w3 = tf.Variable(tf.random.normal([1], mean=0.0, stddev=1.0))
loss_tracker = tf.keras.metrics.Mean(name="loss")
val_loss_tracker = tf.keras.metrics.Mean(name="loss")
accuracy_metric = tf.keras.metrics.BinaryAccuracy()
val_accuracy_metric = tf.keras.metrics.BinaryAccuracy()



for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch+1, epochs))
    for i in tqdm(range(0, len(train_left), batch_size)):
        left_batch = train_left[i:i+batch_size]
        right_batch = train_right[i:i+batch_size]
        labels_batch = train_label[i:i+batch_size]
        p1_batch = train_p1[i:i+batch_size]
        p2_batch = train_p2[i:i+batch_size]
        p3_batch = train_p3[i:i+batch_size]

        # Train on batch
        train_step([left_batch, right_batch], labels_batch, DBRD, optimizer, loss_fn, p1_batch, p2_batch, p3_batch)

    # Evaluate on validation set
    evaluate(val_left, val_right, val_label, DBRD, loss_fn, val_p1, val_p2, val_p3, batch_size)

    # Print training progress
    print("Training accuracy:", accuracy_metric.result().numpy())
    print("Training loss:", loss_tracker.result().numpy())
    print("Validation accuracy:", val_accuracy_metric.result().numpy())
    print("Validation loss:", val_loss_tracker.result().numpy())
    val_loss = val_loss_tracker.result()
    val_acc = val_accuracy_metric.result()
    # Reset metrics at the end of each epoch
    accuracy_metric.reset_states()
    loss_tracker.reset_states()
    val_loss_tracker.reset_states()
    val_accuracy_metric.reset_states()

    wait += 1

    if val_acc > best:
      best = val_acc
      wait = 0
    if val_loss < best_loss:
      best_loss = val_loss
      wait = 0

    if wait >= patience:
      break
    print("Best acc:",best)
    print("Best loss:",best_loss)
    print("Wait:",wait)
print("test")