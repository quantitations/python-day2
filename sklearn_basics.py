
##########################################################################
####################### The scikit-learn package #########################
##########################################################################


# The scikit-learn package has a great deal of functionality that
# streamlines the common tasks and techniques of machine learning.
# I hope these examples give you the flavor of this package,
# but we will only scratch the surface of what it can do.
# https://scikit-learn.org/stable/getting_started.html




# You may have seen the "iris" dataset before.
# It has measurements of sepal lengths and widths and petal lengths and widths
# for three different species of iris plants.
# It's available in sklearn along with many other example datasets.

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
type(X)
X.shape
X[:10, :]
y




# Create an empty model object, then fit it to data

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X, y)

# Now rf is a fitted model and can be used to predict the labels
# for a given set of explanatory variables values.
rf.predict(X)
rf.predict(X) == y
# Performs perfectly on the original data (probably over-fitting!)



# In general, we don't just fit a model on the full dataset.
# Rather, we want to try a variety of techniques to compare them
# and estimate how well they will generalize to future data.
# To this end, we first split data into training and test sets.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.shape
X_test.shape
# By default about a quarter of the data is designated as test data
# but you can use a different proportion, e.g.
# train_test_split(X, y, test_size=.1)

rf.fit(X_train, y_train)
# this "fit" overwrites the original fitting
# however some methods allow for "partial_fit", e.g.
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html

rf.predict(X_test) == y_test
# usually gets a few labels wrong out of 38
# here's a built-in way to calculate the proportion of accurate labels:

from sklearn.metrics import accuracy_score

accuracy_score(rf.predict(X_test), y_test)


# Most machine learning techniques have a number of unspecified hyperparameters
# that determine, for example, how harshly to penalize more complex models.
# Rather than using the default values, it is generally better to
# choose values for the hyperparameters by trying a variety of them
# in cross-validation trials and using the values that work best.
# The following code explores a grid of different combinations of two
# hyperparameters in our random forest classifier and picks the one with
# the smallest average error in 5-fold cross-validation.


from sklearn.model_selection import GridSearchCV

grid = {'n_estimators': [1, 10, 50, 100], 'max_depth': list(range(5, 11))}
rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grid)

rf.fit(X_train, y_train)
accuracy_score(rf.predict(X_test), y_test)

# we can ask which parameter values performed best in cross-validation
# and where therefore used for the fitting:
rf.best_params_


## Now let's try another technique on this data

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression()
logr.fit(X_train, y_train)

accuracy_score(logr.predict(X_test), y_test)

# However, logistic regression tends to work better with standardized explanatory variables

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
logr.fit(scaler.transform(X_train), y_train)

accuracy_score(logr.predict(scaler.transform(X_test)), y_test)

# The transform plus logistic regression sequence can be put together
# into a single "pipeline" object so that we don't have to explicitly
# rescale all future data before using the predict method.

from sklearn.pipeline import make_pipeline

logr = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# treat the pipeline like a model object
logr.fit(X_train, y_train)
logr.predict(X_test)
accuracy_score(logr.predict(X_test), y_test)



# It's remarkable how uniformly we can handle different methods.
# To reinforce this, let's see how both the random forest and logistic
# regression fitting can be put together into a loop.

rf = GridSearchCV(estimator=RandomForestClassifier(),
                  param_grid={'n_estimators': [1, 10, 50, 100], 'max_depth': list(range(5, 11))})

logr = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)


methods = [rf, logr]
for m in methods:
    m.fit(X_train, y_train)
    print(accuracy_score(m.predict(X_test), y_test))


## EXERCISE: Redo the logistic regression, but this time do principal
##           components analysis on the explanatory variables first
##           instead of simply rescaling.
##           Incorporate a grid search that tries keeping 1, 2, 3, and 4 components.
##           This goes beyond what's been presented above, so you'll need
##           to do some web searching and some trial-and-error.
##           But that's an essential part of coding!











####################### SOLUTION ######################


from sklearn.decomposition import PCA

logr = GridSearchCV(estimator=make_pipeline(PCA(), LogisticRegression()),
                    param_grid={'pca__n_components': [1, 2, 3, 4]})

logr.fit(X_train, y_train)
accuracy_score(logr.predict(X_test), y_test)



