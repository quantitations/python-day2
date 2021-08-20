
##########################################################################
####################### The SciPy ecosystem ##############################
##########################################################################



# The "SciPy ecosystem" is a collection of Python packages designed to
# streamline many of the common tasks involved in data science.
# In this workshop, we'll take a look at four of those packages,
# focusing on the features that I find most important and occasionally
# highlighting similarities and differences with how things work in R.


# https://www.scipy.org/





############################ NumPy ##############################


import numpy as np



## ARRAYS

a = np.array([1, 3, 2]) # one-dimensional array, i.e. vector
np.array((1, 3, 2)) # or putting a tuple rather than list - creates same array
a[:2] # subsetting works as usual for Python lists

b = np.array([a, (4, 5, 6)]) # two-dimensional array, i.e. matrix
b
print(b)

b[:, :2] # subsetting all rows and first two columns
b[0, 1] = 8 # set row 0 column 1 entry to 8
print(b)

b[0] # if only one index is provided, it refers to the row(s) and full column slices are taken
b[-1]

# arithmetic operations apply to each entry of an array
b+1
2*b
b**2
2*b - b # subtracts corresponding entries
# (in fact Pandas Series objects are a type of array - they inherit this behavior)

b.sum(axis=0)
b.sum(axis=1)
b.sum() # adds up all the numbers in the matrix

# more sophisticated vector and matrix operations
a.mean() # an equivalent command is "np.mean(a)"
a.var()
a.std()
a.max()
a.argmax()
a.sort() # note that this is an in-place function! it changes the object rather than just returning a sorted version
a

np.dot(b[0], b[1]) # dot product

b.transpose() # equivalently "np.transpose(b)"
np.cov(b) # covariance matrix
b @ b.transpose() # matrix multiplication

np.linalg.svd(b) # singular value decomposition



## MATHEMATICAL FUNCTIONS AND CONSTANTS

np.cos(np.pi)
np.exp(np.log(np.sqrt(25)))


# https://www.numpy.org/devdocs/user/quickstart.html


## EXERCISE 1: Find the variance of each row of b.







############################ Pandas #############################


# The most important aspect of Pandas is that
# it defines a DataFrame class that comes with many
# convenient methods that R users are used to.

import pandas as pd

data_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/anscombe.csv"
z = pd.read_csv(data_url) # import data as a DataFrame, analogous to "read.csv(data_url)"



## BASICS

z.head() # analogous to "head(z)"
z.head(20) # analogous to "head(z, 20)"
z.shape # analogous to "dim(z)"
z.columns # analogous to "names(z)"
z.describe() # analogous to "summary(z)"



### SUBSETTING


## retrieving specified columns

z["x"]
z[["x", "y"]]


## retrieving the rows that satisfy a specified condition

z["x"] > 10 # creates a vector of True/False values telling you whether each row's x-val is > 10
z[z["x"] > 10] # selects the rows that have x-val > 10
z[(z["x"] > 10) & (z["dataset"]=="II")] # selects the rows that have x-val > 10 and dataset equal to "II"
z[(z["x"] > 10) | (z["dataset"]=="II")] # selects the rows that have x-val > 10 or dataset equal to "II"



## subsetting by NAME: loc

# - again, single argument retrieves row, two arguments specify row(s) then column(s)

z.loc[0] # the first row has 0 as its rowname
# But be careful: index names are preserved when new objects are created by subsetting
zz = z.iloc[3:6, :]
zz
zz.iloc[0] # zz has a row with index 0
zz.loc[0] # but its first row isn't named 0
zz.loc[3] # it's name is (still) 3
z.loc[3:6] # also notice that BOTH endpoints are included when using names - very tricky!

z.loc[:4, "x"]
z.loc[:, ["x", "y"]] # same as z[["x", "y"]]



# REMINDER about list pointers and copy()
# Let's see a behavior of lists that can be confusing
zz.loc[3, "x"] = -1
zz
z.head() # it changed the original DataFrame as well!
z.loc[3, "x"] = 9 # let's change it back

zz = z.iloc[3:6, :].copy() # to prevent this issue, we could have explicitly made a copy
zz.loc[3, "x"] = -1 # change the copy
z.head() # check the original
# we've verified that this time, we didn't change the original, only the copy





## EXERCISE 2: Find the sample standard deviation of the y-values in dataset III.
##             (Remember the "describe" function.)








## CALCULATIONS


# Pandas defines a Series object type; each column of a DataFrame is a Series.
# A Series is an "enriched" version of a numpy array
# In fact, a DataFrame is a dictionary
# - each key is a variable name and its value is the corresponding column.
type(z[["x", "y"]])
type(z["x"])

# The head, describe, mean, and additional methods listed here work on Series objects
# or can be easily "applied" to the columns or rows of a DataFrame
# These columns inherit the numpy array methods while adding new ones such as
z["x"].abs()
z["x"].median()

# apply to every column or to every row
z.mean() # analogous to "apply(z, 2, mean)"
z.mean(axis=1) # "apply(z, 1, mean)"
# can also use an "apply" method to specify your own function
z[["x", "y"]].apply(sum)
z[["x", "y"]].apply(sum, axis=1)

# as with arrays, you can easily perform an arithmetic operation to every entry of a Series
3*z["x"] # "3*z$x" in R
# remember, you can't use this syntax with ordinary Python lists!
# you'd use "list comprehension" instead: [3*i for i in z["x"]]

# multiply two vectors, entry by entry
z["x"]*z["y"]
# dot product of x and y:
sum(z["x"]*z["y"])

# splitting up by a categorical variable
# The "groupby" function in Pandas is analogous to "aggregate" or "split" in R.
z.groupby("dataset").describe()

# correlation matrix of each group
z.groupby("dataset").corr()

# groupby creates an iterator, let's see the corresponding list
list(z.groupby("dataset"))

# You can loop through the groups
for groupname, groupdata in z.groupby("dataset"):
    print(groupname)
    print(groupdata)

for groupname, groupdata in z.groupby("dataset")["x"]:
    print(groupname)
    print(groupdata)



## EXERCISE 3: Find the dot product of x and y within each of the four datasets.







############################ SciPy ##############################

# Note that the SciPy package is one of the many packages in the SciPy ecosystem.
# The terminology can be confusing!


## LINEAR ALGEBRA

from scipy.linalg import svd, inv

svd(b) # singular value decomposition, agrees with the NumPy function
inv(b @ b.transpose()) # inverse matrix

# https://docs.scipy.org/doc/scipy/reference/linalg.html



## PROBABILITY DISTRIBUTIONS

from scipy.stats import norm # functions for Normal distributions
from scipy.stats import f # functions for F distributions

# cdf and inverse cdf
2*norm.cdf(-2) # twice the probability of a standard Normal draw being less than -2
norm.ppf(.975) # a standard Normal draw has probability .975 of being less than this value

1-f.cdf(2, 1, 12) # probability that a draw from f_{1, 12} is greater than 2


# simulating random samples

norm.rvs(size=10)
f.rvs(1, 12, size=10)

# For more functions and distributions, see
# https://docs.scipy.org/doc/scipy/reference/stats.html

# https://www.tutorialspoint.com/scipy/scipy_introduction.htm



###########################################################################

## We'll see more from the scipy ecosystem, including the Matplotlib package,
## in plotting.ipynb










############################## SOLUTIONS #####################################


## SOLUTION 1 ##

b.var(axis=1)



## SOLUTION 2 ##

z[z["dataset"]=="III"]
z[z["dataset"]=="III"].describe()
z[z["dataset"]=="III"].describe()["y"]
z[z["dataset"]=="III"].describe()["y"][2]



## SOLUTION 3 ##

for name, group in z.groupby("dataset"):
    print(name, ":", sum(group["x"]*group["y"]))


