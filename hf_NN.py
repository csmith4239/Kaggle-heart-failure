import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load in data.
df = pd.read_csv('heart.csv')

# Split data into dependant and independant variables.
X,y = df[df.columns[:-1]], df[df.columns[-1]]

# Get dummies for categorical variables.
X = pd.get_dummies(X)

# Split the data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = .3, random_state=50)

# Set up and train Regressor NN.
regr = MLPRegressor(random_state=50, max_iter=500, activation='logistic', solver='adam', learning_rate_init=.0001).fit(X_train, y_train)

# Score with no tuning.
print(regr.score(X_test, y_test))

# Set up and train Classifier NN.
clf = MLPClassifier(random_state=50, max_iter=10000, activation='logistic', solver='adam', learning_rate_init=.0001).fit(X_train, y_train)

# Score with no tuning.
print(clf.score(X_test, y_test))

y_pred = clf.predict(X_test)
print((classification_report(y_test, y_pred)))
print(confusion_matrix(y_test, y_pred))

"""
    We see that the Classifier out performs the regressor. This make sense since heart disease is either Y or N.
"""
