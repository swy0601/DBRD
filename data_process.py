import pickle
left = tf.convert_to_tensor(torch.load('/content/drive/MyDrive/data/token200/netbeans_tokken1.pt'))
right = tf.convert_to_tensor(torch.load('/content/drive/MyDrive/data/token200/netbeans_tokken2.pt'))

with open('/content/drive/MyDrive/data/label/nb_l.pkl', 'rb') as f:
    label = pickle.load(f)
label = [0 if i == -1 else i for i in label]
label = tf.convert_to_tensor(label, dtype=tf.int32)
label = np_utils.to_categorical(label, 2)
with open('/content/drive/MyDrive/data/netbeans_normal/netbeans_ct.pkl', 'rb') as f:
    p1 = pickle.load(f)
    p1 = tf.convert_to_tensor(p1)
with open('/content/drive/MyDrive/data/netbeans_normal/netbeans_pr.pkl', 'rb') as f:
    p2 = pickle.load(f)
    p2 = tf.convert_to_tensor(p2)
with open('/content/drive/MyDrive/data/netbeans_normal/netbeans_sv.pkl', 'rb') as f:
    p3 = pickle.load(f)
    p3 = tf.convert_to_tensor(p3)

RANDOM_SEED =22
left = tf.random.shuffle(left, seed=RANDOM_SEED )
right = tf.random.shuffle(right, seed=RANDOM_SEED )
labels = tf.random.shuffle(label, seed=RANDOM_SEED )
p1 = tf.random.shuffle(p1, seed=RANDOM_SEED )
p2 = tf.random.shuffle(p2, seed=RANDOM_SEED )
p3 = tf.random.shuffle(p3, seed=RANDOM_SEED )
data_size = len(labels)
train_size = int(0.8 * data_size)
val_size = int(0.1 * data_size)
test_size = data_size - train_size - val_size

split_left = tf.split(left, [train_size, val_size, test_size], axis=0)
split_right = tf.split(right, [train_size, val_size, test_size], axis=0)
split_p1 = tf.split(p1, [train_size, val_size, test_size], axis=0)
split_p2 = tf.split(p2, [train_size, val_size, test_size], axis=0)
split_p3 = tf.split(p3, [train_size, val_size, test_size], axis=0)
split_labels = tf.split(labels, [train_size, val_size, test_size], axis=0)

train_left = split_left[0]
train_right = split_right[0]
train_label = split_labels[0]
train_p1 = split_p1[0]
train_p2 = split_p2[0]
train_p3 = split_p3[0]

val_left = split_left[1]
val_right = split_right[1]
val_label = split_labels[1]
val_p1 = split_p1[1]
val_p2 = split_p2[1]
val_p3 = split_p3[1]


test_left = split_left[2]
test_right = split_right[2]
test_label = split_labels[2]
test_p1 = split_p1[2]
test_p2 = split_p2[2]
test_p3 = split_p3[2]


#
# import tensorflow as tf
#
# # Combine your data into a single dataset
# dataset = tf.data.Dataset.from_tensor_slices((left, right, label, p1, p2, p3))
#
# # Shuffle the dataset with a seed
# dataset = dataset.shuffle(buffer_size=data_size, seed=RANDOM_SEED)
#
# # Split the dataset into train, validation, and test sets
# train_size = int(0.8 * data_size)
# val_size = int(0.1 * data_size)
# test_size = data_size - train_size - val_size
#
# train_dataset = dataset.take(train_size)
# val_dataset = dataset.skip(train_size).take(val_size)
# test_dataset = dataset.skip(train_size + val_size).take(test_size)
#
# # Now you have your train, validation, and test datasets. You can further process them as needed.
#
# # For example, if you want to separate features and labels:
# train_left, train_right, train_label, train_p1, train_p2, train_p3 = zip(*train_dataset)
# val_left, val_right, val_label, val_p1, val_p2, val_p3 = zip(*val_dataset)
# test_left, test_right, test_label, test_p1, test_p2, test_p3 = zip(*test_dataset)
