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

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers

#%%Merge dispatch data at each time interval
csv_header = 'end_longitude,end_latitude,end_range,origin_longitude,origin_latitude,trip_distance,trip_time,wait_time,pickup_distance,pickup_time_in_mins,timestamp,class'

#set ouput data path and name
csv_out = 'H:\\EAV Taxi Data\\'+scenario+'\\dispatch_output.csv'

#set csv data path
csv_dir = 'H:\\EAV Taxi Data\\'+scenario+'\\Dispatch'

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
df = pd.read_csv('H:\\EAV Taxi Data\\'+scenario+'\\dispatch_output.csv')

df.columns

#convert timestamp to hh:mm relative to the start time of the day
df['timestamp'] = (df['timestamp'] - start_time)/60%1440

used_col = ['end_longitude', 'end_latitude', 'end_range',\
            'origin_longitude', 'origin_latitude', 'trip_distance', 'trip_time', 'wait_time',\
            'pickup_distance', 'pickup_time_in_mins',\
            'timestamp',\
            'class']
#df = df[used_col]

#convert status to factors
#df['status'].value_counts()
#d = {'waiting': 0, 'start charging': 1}
#df['status'] = df['status'].map(d)
#df['status'].value_counts()

#check distribution
df.loc[0]
A = df.describe()
plt.boxplot(df['trip_time'])

"""
#normalize data at the beginning
df['end_longitude'] = (df['end_longitude']-(-74.25))/(-73.5-(-74.25))
df['end_latitude'] = (df['end_latitude']-40.4)/(41.1-40.4)
df['end_range'] = df['end_range']/EV_range
df['origin_longitude'] = (df['origin_longitude']-(-74.25))/(-73.5-(-74.25))
df['origin_latitude'] = (df['origin_latitude']-40.4)/(41.1-40.4)

df['trip_distance'] = (df['trip_distance']-1)/(3-1)
df['trip_time'] = (df['trip_time']-7)/(18-7)

df['wait_time'] = df['wait_time']/max_wait_time
df['pickup_distance'] = df['pickup_distance']/max_pickup_dist
df['pickup_time_in_mins'] = df['pickup_time_in_mins']/max_pickup_time
df['timestamp'] = df['timestamp']/1440
"""

#%%prepare data
X = df[used_col[:-1]] #other than class
y = df['class']
del df

#class imbalance
y.value_counts()
y.value_counts()[1]/y.count()

#over-sampling, select with replacement
ros = RandomOverSampler(random_state=1991)
X_sampled, y_sampled = ros.fit_sample(X, y)
np.unique(y_sampled, return_counts=True)
"""
#no sampling
X_sampled = X.values
y_sampled = y.values

#over-sampling, SMOTE
smote = SMOTE(ratio='minority', random_state=1991)
X_sampled, y_sampled = smote.fit_sample(X, y)
np.unique(y_sampled, return_counts=True)

#under-sampling
rus = RandomUnderSampler(random_state=1991)
X_sampled, y_sampled = rus.fit_sample(X, y)
np.unique(y_sampled, return_counts=True)

#combine over-sampling and under-sampling, SMOTEENN
smote_enn = SMOTEENN(random_state=1991)
X_sampled, y_sampled = smote_enn.fit_sample(X, y)
np.unique(y_sampled, return_counts=True)

#combine over-sampling and under-sampling, SMOTETomek
smote_tomek = SMOTETomek(random_state=1991)
X_resampled, y_resampled = smote_tomek.fit_sample(X, y)
np.unique(y_sampled, return_counts=True)
"""
#split data
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.1, random_state=1992)
del X_sampled, y_sampled

#scale data, method 3, used as the final scale method
sc = preprocessing.RobustScaler()
#sc = preprocessing.MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sc.center_
sc.scale_
#save sc for later use
sc_pickle = {}
sc_pickle['sc'] = sc
pickle.dump(sc_pickle, open("H:\\EAV Taxi Data\\"+scenario+"\\sc_pickle.p", "wb"))

#shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state=611)

#scale sampled data, for hyperparameter tuning only
#sc = preprocessing.RobustScaler()
#X_sampled_sc = sc.fit_transform(X_sampled)
#y_sampled_sc = y_sampled

#%%deep learning models

#hyperparameter tuning
tuning = '128-64-8-256'

#ANN
model = Sequential()
model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
#model.add(Dropout(0.1)) #fraction of the input units to drop
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.1))
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
pickle.dump(history_pickle, open("H:\\EAV Taxi Data\\"+scenario+"\\history_pickle_"+tuning+".p", "wb"))

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
plt.tight_layout()
plt.show()
fig.savefig("H:\\EAV Taxi Data\\"+scenario+"\\accuracy_"+tuning+".jpg", dpi=300)

fig = plt.figure(figsize=(5, 3))
plt.plot(epoch, train_loss)
plt.plot(epoch, val_loss)
plt.legend(['training set', 'validation set'], loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()
fig.savefig("H:\\EAV Taxi Data\\"+scenario+"\\loss_"+tuning+".jpg", dpi=300)

#save model
model.save('H:\\EAV Taxi Data\\'+scenario+'\\ann_model_'+tuning+'.h5')

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








#%%machine learning models
#decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
confusion_matrix(y_test, prediction)
accuracy_score(y_test, prediction)
clf.score(X_test, y_test)

#use cross validation
from sklearn.model_selection import cross_val_score
clf = tree.DecisionTreeClassifier()
fold_scores = cross_val_score(clf, X_scaled, y_os, cv=5)
print (fold_scores)
print (np.mean(fold_scores))

#display decesion tree
from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data, feature_names=X.columns)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png()) 

#Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)

clf = RandomForestClassifier(n_estimators=10)
fold_scores = cross_val_score(clf, X_scaled, y_os, cv=5)
print (fold_scores)
print (np.mean(fold_scores))

#scikit-learn NN
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = clf.fit(X_scaled, y_os)
prediction = clf.predict(X_scaled)
confusion_matrix(y_os, prediction)
accuracy_score(y_os, prediction)

#NB
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)

#SVM
from sklearn import svm
svc = svm.SVC(kernel='linear', C=1.0).fit(X_train, y_train)
svc.score(X_test, y_test)

#%%not useful code

#Merge dispatch data at each time interval
#could be very slow
folder = "H:\\EAV Taxi Data\\"+scenario+"\\Dispatch"
os.chdir(folder) #os.chdir() is very useful
combined_file = pd.DataFrame([])
for counter, file in enumerate(glob.glob('*.csv')):
    namedf = pd.read_csv(file)
    #delete too far taxi-request pairs
    #namedf = namedf[namedf['pickup_time_in_mins'] <= max_pickup_time]
    combined_file = combined_file.append(namedf)
      
combined_file.to_csv('H:\\EAV Taxi Data\\'+scenario+'\\dispatch_output.csv', index=False)
combined_file = combined_file.reset_index(drop=True)






#scale data, method 1
X_train = preprocessing.scale(X_train)
X_train.mean(axis=0)
X_train.std(axis=0)

#scale data, methos 2
sc = preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sc.mean_
sc.scale_
#save sc for later use
sc_pickle = {}
sc_pickle['sc'] = sc
pickle.dump(sc_pickle, open("H:\\EAV Taxi Data\\sc_pickle.p", "wb"))





model = Sequential()
model.add(Dense(64, input_dim=X.shape[1]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(8))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))

history = model.fit(X_sampled, y_sampled,
          epochs=100, verbose=2,
          batch_size=128,
          validation_split=0.2)
model.summary()







#tune optimizer
def create_model(optimizer):
    model = Sequential()
    model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

np.random.seed(2015)
model = KerasClassifier(build_fn=create_model, epochs=1, batch_size=128, verbose=0)
#define the grid search parameters
optimizer = ['sgd', 'adam']
param_grid = dict(optimizer=optimizer)
scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_sampled_sc, y_sampled_sc)

#summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))