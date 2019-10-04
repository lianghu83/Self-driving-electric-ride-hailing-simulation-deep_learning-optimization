"""
Machine learning algorithms to learn the optimal dispatch.
"""

from Parameters import max_pickup_time, scenario, start_time, EV_range, max_wait_time, max_pickup_dist

import pandas as pd
import numpy as np
import pickle
import shutil
#from os import listdir
import glob
import os 
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers

#%%Merge dispatch data at each time interval
csv_header = 'end_longitude,end_latitude,end_range,origin_longitude,origin_latitude,trip_distance,trip_time,wait_time,pickup_distance,pickup_time_in_mins,timestamp,class'

#set ouput data path and name
csv_out = 'C:\\EAV Taxi Data\\'+'10-16_op_R2'+'\\dispatch_output.csv'

#set csv data path
csv_dir = 'C:\\EAV Taxi Data\\'+'10-16_op_R2'+'\\Dispatch'

dir_tree = os.walk(csv_dir)
for dirpath, dirnames, filenames in dir_tree:
   pass

csv_list = []
for file in filenames:
   if file.endswith('.csv'):
      csv_list.append(file)
csv_list = [csv_dir+'\\'+x for x in csv_list]

csv_merge = open(csv_out, 'w')
csv_merge.write(csv_header)
csv_merge.write('\n')

for file in csv_list:
   csv_in = open(file)
   for line in csv_in:
      if line.startswith(csv_header):
         continue
      csv_merge.write(line)
   csv_in.close()

csv_merge.close()

print('Verify consolidated CSV file : ' + csv_out)

#%%prepare learning data
df = pd.read_csv('C:\\EAV Taxi Data\\'+'9_10-12_R2'+'\\dispatch_output.csv')

df.columns

#convert timestamp to hh:mm relative to the start time of the day
df['timestamp'] = (df['timestamp'] - 1378785600)/60%1440 #2013-9-10 00:00:00

used_col = ['end_longitude', 'end_latitude', 'end_range',\
            'origin_longitude', 'origin_latitude', 'trip_distance', 'trip_time', 'wait_time',\
            'pickup_distance', 'pickup_time_in_mins',\
            'timestamp',\
            'class']

#%%prepare data
X = df[used_col[:-1]] #other than class
y = df['class']

#class imbalance
y.value_counts()
y.value_counts()[1]/y.count()

#scale data, method 3, used as the final scale method
sc = preprocessing.RobustScaler()
X = sc.fit_transform(X)
sc.center_
sc.scale_
#save sc for later use
sc_pickle = {}
sc_pickle['sc'] = sc
pickle.dump(sc_pickle, open("C:\\EAV Taxi Data\\"+"9_10-12_R2"+"\\sc_pickle.p", "wb"))

#shuffle data
X, y = shuffle(X, y, random_state=83)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1220, stratify=y) #used to be 1992, 611, 
y_train.value_counts()[1]/y_train.count()
y_test.value_counts()[1]/y_test.count()

#over-sampling, select with replacement
ros = RandomOverSampler(random_state=1991)
X_train, y_train = ros.fit_sample(X_train, y_train)
np.unique(y_train, return_counts=True)

#shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state=611)

#clear memory
del df, X, y

#%%another day's dispatch data
valid = pd.read_csv('C:\\EAV Taxi Data\\'+'10-16_op_R2'+'\\dispatch_output.csv')
#valid = valid.sample(frac=0.1)
valid['timestamp'] = (valid['timestamp'] - 1381896000)/60%1440 #2013-10-16 00:00:00
X_valid = valid[used_col[:-1]] #other than class
y_valid = valid['class']
y_valid.value_counts()[1]/y_valid.count()
sc_pickle = pickle.load(open("C:\\EAV Taxi Data\\9_10-12_R2"+"\\sc_pickle.p", "rb" ))
sc = sc_pickle["sc"]
X_valid = sc.transform(X_valid)

#%%deep learning models

#hyperparameter tuning
tuning = '128-64-8-256'

#ANN
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2)) #fraction of the input units to drop
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #since it is a binary classification, use sigmoid
#adam = optimizers.Adam(lr=0.005)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train,
          epochs=100, verbose=2,
          batch_size=256,
          validation_data=(X_test, y_test))
model.summary()

#save history
history_pickle = {}
history_pickle['history'] = history
pickle.dump(history_pickle, open("C:\\EAV Taxi Data\\"+"9_10-12_R2"+"\\history_pickle_"+tuning+".p", "wb"))

#save model
model.save('C:\\EAV Taxi Data\\'+"9_10-12_R2"+'\\ann_model_'+tuning+'.h5')



#plot training process
epoch = np.array(history.epoch)+1
train_loss = history.history['loss']
train_acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']

fig = plt.figure(figsize=(5, 3))
plt.plot(epoch, train_acc)
plt.plot(epoch, val_acc)
plt.legend(['training set', 'validation set'], loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim((0.85,0.93))
plt.tight_layout()
plt.show()
fig.savefig("C:\\EAV Taxi Data\\"+"9_10-12_R2"+"\\accuracy_"+tuning+".jpg", dpi=300)

fig = plt.figure(figsize=(5, 3))
plt.plot(epoch, train_loss)
plt.plot(epoch, val_loss)
plt.legend(['training set', 'validation set'], loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()
fig.savefig("C:\\EAV Taxi Data\\"+"9_10-12_R2"+"\\loss_"+tuning+".jpg", dpi=300)



#evaluate model
y_pred = model.predict_classes(X_test)
accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")

print(classification_report(y_test, y_pred))

scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#the output probability is for the class=1, which is pickup
y_pred_p = model.predict(X_test)

#on the other day's dataset
#load ANN model if necessary
from keras.models import load_model
model = load_model('C:\\EAV Taxi Data\\9_10-12_R2'+'\\ann_model_128-64-8-256.h5')
y_pred = model.predict_classes(X_valid)
accuracy_score(y_valid, y_pred)

cm = confusion_matrix(y_valid, y_pred)
sns.heatmap(cm, annot=True, fmt="d")

print(classification_report(y_valid, y_pred))



#%%machine learning models

#logistic regression
clf = LogisticRegression()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_valid)
accuracy_score(y_valid, y_pred)
print(classification_report(y_valid, y_pred))

#save model
pickle.dump(clf, open("H:\\EAV Taxi Data\\"+"9_10-12_R1"+"\\logistic_regression.sav", "wb"))





#decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_valid)
accuracy_score(y_valid, y_pred)
print(classification_report(y_valid, y_pred))



#%%use old model on one day's dispatch data
from keras.models import load_model
ml_model = load_model('H:\\EAV Taxi Data\\9_10-12'+'\\ann_model_128-64-8-256.h5')
import pickle
sc_pickle = pickle.load(open("H:\\EAV Taxi Data\\9_10-12"+"\\sc_pickle.p", "rb" ))
sc = sc_pickle["sc"]

y_pred = ml_model.predict_classes(X_valid)
accuracy_score(y_valid, y_pred)

cm = confusion_matrix(y_valid, y_pred)
sns.heatmap(cm, annot=True, fmt="d")

print(classification_report(y_valid, y_pred))









