
# coding: utf-8

# In[2]:
import keras
import numpy as np
import pandas as pd
from keras import models
from keras.models import Sequential
from keras.models import model_from_json
import os
# In[3]:
from keras import callbacks
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.externals import joblib


# In[5]:

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[23]:

# load dataset
dataframe = pd.read_csv("D:\PROJECT\compiled2.csv", header=None)
dataframe.dropna(axis=1, how='all')
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:18].astype(float)
Y = dataset[:,18]


# In[15]:

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(64, input_dim=18, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return model

model=create_baseline();

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')


# In[20]:

pipe = pipeline.Pipeline([
    ('rescale', preprocessing.StandardScaler()),
    ('nn', KerasClassifier(build_fn=model, nb_epoch=10, batch_size=128,
                           validation_split=0.2, callbacks=[early_stopping]))
])




# In[21]:

#evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=model, epochs=1, batch_size=4, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



#keras.utils.plot_model(   , to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')

#model = create_baseline()
#model.fit(x=X, y=Y, verbose=1, validation_split=0.1, epochs=100)
#evaluate baseline model with standardized dataset



#pipe.fit(X.values, encoded_Y.values)

directory = os.path.dirname(os.path.realpath(__file__))
model_step = pipe.steps.pop(-1)[1]
joblib.dump(pipe, os.path.join(directory, 'pipeline.pkl'))

#saving

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


np.random.seed(seed)
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
#pipeline = Pipeline(estimators)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
#print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]: