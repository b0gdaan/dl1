# https://www.kaggle.com/competitions/titanic
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random
import seaborn

train_data = pd.read_csv("train.csv")
# print(train_data.head())

test_data = pd.read_csv("test.csv")
# print(test_data.head())

gender_map = {'male': 1, 'female': 0}
train_data['Sex'] = train_data['Sex'].map(gender_map)
test_data['Sex'] = test_data['Sex'].map(gender_map)

# print(train_data['Embarked'].unique())
embarked_map = {'Q': 0, 'S': 0.5, 'C': 1}
train_data['Embarked'] = train_data['Embarked'].map(embarked_map)
test_data['Embarked'] = test_data['Embarked'].map(embarked_map)
average = train_data['Embarked'].mean()
# print(average)
train_data['Embarked'] = train_data['Embarked'].fillna(0.55)
test_data['Embarked'] = test_data['Embarked'].fillna(0.55)
# print(train_data['Embarked'].unique())

# print(train_data['Age'].mean())
train_data['Age'] = train_data['Age'].fillna(29.7)

# print(train_data['Pclass'].isnull().any())
# print(train_data['Sex'].isnull().any())
# print(train_data['Age'].isnull().any())
# print(train_data['SibSp'].isnull().any())
# print(train_data['Parch'].isnull().any())
# print(train_data['Fare'].isnull().any())
# print(train_data['Embarked'].isnull().any())

for column in ["Pclass", "Age",  "Fare"]:
    train_data[column] = StandardScaler().fit_transform(train_data[[column]])
    test_data[column] = StandardScaler().fit_transform(test_data[[column]])


for column in ["SibSp", "Parch"]:
    train_data[column] = StandardScaler().fit_transform(train_data[[column]])
    test_data[column] = StandardScaler().fit_transform(test_data[[column]])

y = train_data["Survived"]
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

random.seed(42)
np.random.seed(42)

def baseline_model():
    model = Sequential()

    model.add(Dense(10, input_dim=len(features), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
# print(class_weight_dict)

model = baseline_model()
history = model.fit(X_train, y_train, epochs=350, batch_size=15, validation_data=(X_validation, y_validation),
                    callbacks=[early_stopping], class_weight=class_weight_dict)

# plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(['Train', 'Vadidation'])
plt.show()

# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'Vadidation'])
plt.show()

model.fit(X, y, epochs=75, batch_size=20, callbacks=[early_stopping], class_weight=class_weight_dict)
predictions = (model.predict(X_test) > 0.5).astype(int).flatten()

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)