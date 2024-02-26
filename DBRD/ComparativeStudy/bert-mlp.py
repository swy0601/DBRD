

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import re
warnings.filterwarnings('ignore')

"""##**Importing the dataset from Drive**"""

# from google.colab import drive
# drive.mount('/content/gdrive')

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
print("读取bert compeleted")
#Thunderbird




"""##**Loading the Pre-trained BERT model**"""
print("读取csv compeleted")
import time
start = time.time()

# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
# tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# model = model_class.from_pretrained(pretrained_weights)


batch_31=df1[:500]
batch_32=df2[:500]
df3 = pd.concat([batch_31,batch_32], ignore_index=True)
batch_41=df1[500:1000]
batch_42=df2[500:1000]
df4 = pd.concat([batch_41,batch_42], ignore_index=True)
batch_51=df1[1000:1500]
batch_52=df2[1000:1500]
df5 = pd.concat([batch_51,batch_52], ignore_index=True)
batch_61=df1[1500:2000]
batch_62=df2[1500:2000]
df6 = pd.concat([batch_61,batch_62], ignore_index=True)
batch_71=df1[2000:2500]
batch_72=df2[2000:2500]
df7 = pd.concat([batch_71,batch_72], ignore_index=True)
batch_81=df1[2500:3000]
batch_82=df2[2500:3000]
df8 = pd.concat([batch_81,batch_82], ignore_index=True)
batch_91=df1[3000:3486]
batch_92=df2[3000:3486]
df9 = pd.concat([batch_91,batch_92], ignore_index=True)


#Testing
batch_101=df1[3486:3900]
batch_102=df2[3486:3900]
df10 = pd.concat([batch_101,batch_102], ignore_index=True)
batch_111=df1[3900:4338]
batch_112=df2[3900:4374]
df11 = pd.concat([batch_111,batch_112], ignore_index=True)



### **_get_segments3**
# """
#
def _get_segments3(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = False
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        #print(token)
        if token == 102:
            #if first_sep:
                #first_sep = False
            #else:
           current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))
#
# """#**df3**


pair3= df3['Title1'] + df3['Description1']+ [" [SEP] "] + df3['Title2'] + df3['Description2']
tokenized3 = pair3.apply((lambda x: tokenizer.encode(x, add_special_tokens=True,truncation=True, max_length=300)))

max_len3 = 0                 # padding all lists to the same size
for i in tokenized3.values:
    if len(i) > max_len3:
        max_len3 = len(i)
max_len3 =300
padded3 = np.array([i + [0]*(max_len3-len(i)) for i in tokenized3.values])

np.array(padded3).shape

attention_mask3 = np.where(padded3 != 0, 1, 0)
attention_mask3.shape
input_ids3 = torch.tensor(padded3)
attention_mask3 = torch.tensor(attention_mask3)
input_segments3= np.array([_get_segments3(token, max_len3)for token in tokenized3.values])
token_type_ids3 = torch.tensor(input_segments3)
input_segments3 = torch.tensor(input_segments3)

with torch.no_grad():
    last_hidden_states3 = model(input_ids3, attention_mask=attention_mask3, token_type_ids=input_segments3)    # <<< 600 rows only !!!
features3 = last_hidden_states3[0][:,0,:].numpy()
features3
print('feature3 is already')
"""#**df4**"""

pair4=df4['Title1'] + df4['Description1']+ [" [SEP] "] + df4['Title2']  + df4['Description2']
tokenized4 = pair4.apply((lambda x: tokenizer.encode(x, add_special_tokens=True,truncation=True, max_length=300)))
max_len4 = 0                 # padding all lists to the same size
for i in tokenized4.values:
    if len(i) > max_len4:
        max_len4 = len(i)
max_len4 =300
padded4 = np.array([i + [0]*(max_len4-len(i)) for i in tokenized4.values])

np.array(padded4).shape

attention_mask4 = np.where(padded4 != 0, 1, 0)
attention_mask4.shape
input_ids4 = torch.tensor(padded4)
attention_mask4 = torch.tensor(attention_mask4)
input_segments4= np.array([_get_segments3(token, max_len4)for token in tokenized4.values])
token_type_ids4 = torch.tensor(input_segments4)
input_segments4 = torch.tensor(input_segments4)

with torch.no_grad():
    last_hidden_states4 = model(input_ids4, attention_mask=attention_mask4, token_type_ids=input_segments4)
features4 = last_hidden_states4[0][:,0,:].numpy()
features4
print('feature4 is already')
"""#**df5**"""

pair5=df5['Title1'] + df5['Description1']+ [" [SEP] "] + df5['Title2'] + df5['Description2']
tokenized5 = pair5.apply((lambda x: tokenizer.encode(x, add_special_tokens=True,truncation=True, max_length=300)))

"""##**Padding**"""

max_len5 = 0                 # padding all lists to the same size
for i in tokenized5.values:
    if len(i) > max_len5:
        max_len5 = len(i)

max_len5 =300
padded5 = np.array([i + [0]*(max_len5-len(i)) for i in tokenized5.values])

np.array(padded5).shape        # Dimensions of the padded variable

"""##**Masking**"""

attention_mask5 = np.where(padded5 != 0, 1, 0)
attention_mask5.shape
input_ids5 = torch.tensor(padded5)
attention_mask5 = torch.tensor(attention_mask5)

"""##**Running the `model()` function through BERT**"""

input_segments5= np.array([_get_segments3(token, max_len5)for token in tokenized5.values])
token_type_ids5 = torch.tensor(input_segments5)
input_segments5 = torch.tensor(input_segments5)

with torch.no_grad():
    last_hidden_states5 = model(input_ids5, attention_mask=attention_mask5, token_type_ids=input_segments5)    # <<< 600 rows only !!!

"""##**Slicing the part of the output of BERT : [cls]**"""

features5 = last_hidden_states5[0][:,0,:].numpy()
features5
print('feature5 is already')
"""#**df6**"""

pair6=df6['Title1'] + df6['Description1']+ [" [SEP] "] + df6['Title2'] + df6['Description2']
tokenized6 = pair6.apply((lambda x: tokenizer.encode(x, add_special_tokens=True,truncation=True, max_length=300)))

max_len6 = 0                 # padding all lists to the same size
for i in tokenized6.values:
    if len(i) > max_len6:
        max_len6 = len(i)

max_len6=300
padded6 = np.array([i + [0]*(max_len6-len(i)) for i in tokenized6.values])

np.array(padded6).shape        # Dimensions of the padded variable

attention_mask6 = np.where(padded6 != 0, 1, 0)
attention_mask6.shape
input_ids6 = torch.tensor(padded6)
attention_mask6 = torch.tensor(attention_mask6)
input_segments6= np.array([_get_segments3(token, max_len6)for token in tokenized6.values])
token_type_ids6 = torch.tensor(input_segments6)
input_segments6 = torch.tensor(input_segments6)

with torch.no_grad():
    last_hidden_states6 = model(input_ids6, attention_mask=attention_mask6, token_type_ids=input_segments6)
features6 = last_hidden_states6[0][:,0,:].numpy()
features6
print('feature6 is already')
"""#**df7**"""

pair7=df7['Title1'] + df7['Description1']+ [" [SEP] "] + df7['Title2'] + df7['Description2']
tokenized7 = pair7.apply((lambda x: tokenizer.encode(x, add_special_tokens=True,truncation=True, max_length=300)))

max_len7 = 0                 # padding all lists to the same size
for i in tokenized7.values:
    if len(i) > max_len7:
        max_len7 = len(i)

max_len7=300
padded7 = np.array([i + [0]*(max_len7-len(i)) for i in tokenized7.values])

np.array(padded7).shape        # Dimensions of the padded variable

attention_mask7 = np.where(padded7 != 0, 1, 0)
attention_mask7.shape
input_ids7 = torch.tensor(padded7)
attention_mask7 = torch.tensor(attention_mask7)
input_segments7= np.array([_get_segments3(token, max_len7)for token in tokenized7.values])
token_type_ids7 = torch.tensor(input_segments7)
input_segments7 = torch.tensor(input_segments7)

with torch.no_grad():
    last_hidden_states7 = model(input_ids7, attention_mask=attention_mask7, token_type_ids=input_segments7)
features7 = last_hidden_states7[0][:,0,:].numpy()
features7
print('feature7 is already')
"""#**df8**"""

pair8=df8['Title1'] + df8['Description1']+ [" [SEP] "] + df8['Title2'] + df8['Description2']
tokenized8 = pair8.apply((lambda x: tokenizer.encode(x, add_special_tokens=True,truncation=True, max_length=300)))

max_len8 = 0                 # padding all lists to the same size
for i in tokenized8.values:
    if len(i) > max_len8:
        max_len8 = len(i)
max_len8=300
padded8 = np.array([i + [0]*(max_len8-len(i)) for i in tokenized8.values])

np.array(padded8).shape        # Dimensions of the padded variable


attention_mask8 = np.where(padded8 != 0, 1, 0)
attention_mask8.shape
input_ids8 = torch.tensor(padded8)
attention_mask8 = torch.tensor(attention_mask8)
input_segments8= np.array([_get_segments3(token, max_len8)for token in tokenized8.values])
token_type_ids8 = torch.tensor(input_segments8)
input_segments8 = torch.tensor(input_segments8)

with torch.no_grad():
    last_hidden_states8 = model(input_ids8, attention_mask=attention_mask8, token_type_ids=input_segments8)
features8 = last_hidden_states8[0][:,0,:].numpy()
features8
print('feature8 is already')
"""#**df9**"""

pair9=df9['Title1'] + df9['Description1']+ [" [SEP] "] + df9['Title2'] + df9['Description2']
tokenized9 = pair9.apply((lambda x: tokenizer.encode(x, add_special_tokens=True,truncation=True, max_length=300)))

max_len9 = 0                 # padding all lists to the same size
for i in tokenized9.values:
    if len(i) > max_len9:
        max_len9 = len(i)
max_len9=300
padded9 = np.array([i + [0]*(max_len9-len(i)) for i in tokenized9.values])

np.array(padded9).shape        # Dimensions of the padded variable

attention_mask9 = np.where(padded9 != 0, 1, 0)
attention_mask9.shape
input_ids9 = torch.tensor(padded9)
attention_mask9 = torch.tensor(attention_mask9)
input_segments9= np.array([_get_segments3(token, max_len9)for token in tokenized9.values])
token_type_ids9 = torch.tensor(input_segments9)
input_segments9 = torch.tensor(input_segments9)

with torch.no_grad():
    last_hidden_states9 = model(input_ids9, attention_mask=attention_mask9, token_type_ids=input_segments9)
features9 = last_hidden_states9[0][:,0,:].numpy()
features9
print('feature9 is already')
"""#**df10**"""

pair10=df10['Title1'] + df10['Description1']+ [" [SEP] "] + df10['Title2'] + df10['Description2']
tokenized10 = pair10.apply((lambda x: tokenizer.encode(x, add_special_tokens=True,truncation=True, max_length=300)))
max_len10 = 0                 # padding all lists to the same size
for i in tokenized10.values:
    if len(i) > max_len10:
        max_len10 = len(i)
max_len10=300
padded10 = np.array([i + [0]*(max_len10-len(i)) for i in tokenized10.values])

np.array(padded10).shape        # Dimensions of the padded variable

attention_mask10 = np.where(padded10 != 0, 1, 0)
attention_mask10.shape
input_ids10 = torch.tensor(padded10)
attention_mask10 = torch.tensor(attention_mask10)
input_segments10= np.array([_get_segments3(token, max_len10)for token in tokenized10.values])
token_type_ids10 = torch.tensor(input_segments10)
input_segments10 = torch.tensor(input_segments10)

with torch.no_grad():
    last_hidden_states10 = model(input_ids10, attention_mask=attention_mask10, token_type_ids=input_segments10)
features10 = last_hidden_states10[0][:,0,:].numpy()
features10
print('feature10 is already')

"""#**df11**"""

pair11=df11['Title1'] + df11['Description1']+ [" [SEP] "] + df11['Title2'] + df11['Description2']
tokenized11 = pair11.apply((lambda x: tokenizer.encode(x, add_special_tokens=True,truncation=True, max_length=300)))

max_len11 = 0                 # padding all lists to the same size
for i in tokenized11.values:
    if len(i) > max_len11:
        max_len11 = len(i)
max_len11=300
padded11 = np.array([i + [0]*(max_len11-len(i)) for i in tokenized11.values])

np.array(padded11).shape        # Dimensions of the padded variable

attention_mask11 = np.where(padded11 != 0, 1, 0)
attention_mask11.shape
input_ids11 = torch.tensor(padded11)
attention_mask11 = torch.tensor(attention_mask11)
input_segments11= np.array([_get_segments3(token, max_len11)for token in tokenized11.values])
token_type_ids11 = torch.tensor(input_segments11)
input_segments11 = torch.tensor(input_segments11)


with torch.no_grad():
    last_hidden_states11 = model(input_ids11, attention_mask=attention_mask11, token_type_ids=input_segments11)
features11 = last_hidden_states11[0][:,0,:].numpy()
features11

print('feature11 is already')

"""#**Classification**"""

features=np.concatenate([features3,features4,features5,features6,features7,features8,features9,features10,features11])
#,features12,features13,features14,features15,features16,features17,features18,features19,features20,features21,features22,features23,features24,features25,features26,features27,features29, features30, features31]
#features=np.concatenate([features3,features4,features5,features6,features7,features8,features9,features10,features11])

features.shape

Total = pd.concat([df3,df4,df5,df6,df7,df8,df9,df10,df11], ignore_index=True)



labels =Total['Label']
labels

"""hold out """

# for thunderbird
]


print('model begin training')
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Generate your dataset or load your data
# train_features, test_features, train_labels, test_labels = make_classification(n_samples=100, n_features=20, random_state=42)

# Define the MLPClassifier and the parameter space
mlp = MLPClassifier(max_iter=100)
parameter_space = {
    'hidden_layer_sizes': [(50, 100, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

# Split your data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform k-fold cross-validation
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(train_features, train_labels)

# Best parameter set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)

# Evaluate on the test set
y_pred = clf.predict(test_features)

print('Results on the test set:')
print(classification_report(test_labels, y_pred))
print(confusion_matrix(test_labels, y_pred))
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 假设你有一个特征矩阵X和目标向量y

# 创建一个k折交叉验证对象，这里k=5
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
#
# # 创建一个分类器（这里以随机森林为例）
# classifier = RandomForestClassifier()
#
# # 使用cross_val_score函数执行k折交叉验证并得到性能指标（比如准确率）
# scores = cross_val_score(classifier, X, y, cv=kf)
#
# # 输出每次交叉验证的性能和平均性能
# print("每次交叉验证的准确率:", scores)
print("平均准确率:", scores.mean())

