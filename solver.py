# %%
import preprocessor as pp
import pandas as pd
import numpy as np


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn import tree

# %%
# BASIC DATA PREPROCESSING
Y, X = pp.basic()
Y = pd.get_dummies(Y, prefix="Survived").astype('int')
# %%
# Split into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, shuffle=False)
# %%
# BASIC NEURAL NETWORK
nn = Sequential([
    Dense(20, activation = 'relu', input_dim = 10),
    Dropout(.1),
    Dense(10, activation = 'relu'),
    Dropout(.1),
    Dense(5, activation = 'relu'),
    Dropout(.1),
    Dense(2, activation = 'softmax')
])

nn.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
# %%
# Train 
history = nn.fit(X_train, Y_train, validation_split=0.15, epochs=200)
# %%
# Predict 
y_pred = nn.predict(X_test)
# Process this into actually readable data
results = []
right_guesses = 0
for index, value in enumerate(y_pred):
    predicted = np.argmax(value)
    expected = np.argmax(Y_test.iloc[index])
    if predicted == expected:
        right_guesses += 1
    results.append("P: " + str(predicted) + " E: " + str(expected) +  " " + str(predicted==expected))
acc_nnb = right_guesses/len(results) * 100
# %%
# BASIC DECISION TREE
d_tree = tree.DecisionTreeClassifier(criterion="entropy")
d_tree.fit(X_train, Y_train)
# %%
y_pred_tree = d_tree.predict(X_test)
results_tree = []
right_guesses = 0
for index, value in enumerate(y_pred_tree):
    predicted = np.argmax(value)
    expected = np.argmax(Y_test.iloc[index])
    if predicted == expected:
        right_guesses += 1
    results_tree.append("P: " + str(predicted) + " E: " + str(expected) +  " " + str(predicted==expected))
acc_dtb = right_guesses/len(results_tree) * 100

# %%
# ADVANCED DATA PREPROCESSING
Y, X = pp.advanced()
Y = pd.get_dummies(Y, prefix="Survived").astype('int')
# %%
# Split into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, shuffle=False)
# %%
# ADVANCED DECISION TREE
d_tree = tree.DecisionTreeClassifier(criterion="entropy")
d_tree.fit(X_train, Y_train)
y_pred_tree = d_tree.predict(X_test)
results_tree = []
right_guesses = 0
for index, value in enumerate(y_pred_tree):
    predicted = np.argmax(value)
    expected = np.argmax(Y_test.iloc[index])
    if predicted == expected:
        right_guesses += 1
    results_tree.append("P: " + str(predicted) + " E: " + str(expected) +  " " + str(predicted==expected))
acc_dta = right_guesses/len(results_tree) * 100
# %%
# ADVANCED NEURAL NETWORK 
nn = Sequential([
    Dense(20, activation = 'relu', input_dim = 27),
    Dropout(.1),
    Dense(10, activation = 'relu'),
    Dropout(.1),
    Dense(5, activation = 'relu'),
    Dropout(.1),
    Dense(2, activation = 'softmax')
])

nn.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
# %%
# Train 
history = nn.fit(X_train, Y_train, validation_split=0.15, epochs=200)
# %%
# Predict 
y_pred = nn.predict(X_test)
# Process this into actually readable data
results = []
right_guesses = 0
for index, value in enumerate(y_pred):
    predicted = np.argmax(value)
    expected = np.argmax(Y_test.iloc[index])
    if predicted == expected:
        right_guesses += 1
    results.append("P: " + str(predicted) + " E: " + str(expected) +  " " + str(predicted==expected))
acc_nna = right_guesses/len(results) * 100
# %%
print("Basic data preprocessing: ")
print("Neural network accuracy: " + str(acc_nnb))
print("Decision tree accuracy: " + str(acc_dtb))
print("More advanced data preprocessing: ")
print("Neural network accuracy:  " + str(acc_nna))
print("Decision tree accuracy:" + str(acc_dta))

# %%
