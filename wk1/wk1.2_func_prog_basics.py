###########################################
# Week 1.2: Functional Programming Basics #
###########################################


################################
# Checking System Requirements #
################################

if ('sc' in locals() or 'sc' in globals()):
    print('''<<<<<!!!!! It seems that you are running in a IBM Watson Studio Apache
    Spark Notebook. Please run it in an IBM Watson Studio Default Runtime
    (without Apache Spark) !!!!! >>>>>''')


#######################
# Importing Libraries #
#######################

import findspark
findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


# initializing a spark object
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()


######################################
# Implementing Conditionals in Spark #
######################################

# function to test greater than 50
def gt50(i):
    if i > 50:
        return True
    else:
        return False

print(gt50(4)) # false
print(gt50(51)) # true

# using lambda to declare the same function
gt50 = lambda i:  i> 50

# testing
print(gt50(4))
print(gt50(51))
# checks out

# using the shuffle functionality
from random import shuffle
l = list(range(100)) # declaring chronological list

# list is being shuffled randomly
shuffle(l)
rdd = sc.parallelize(l)

# using conditional test function to subset qualified cases, and printing cases
print(rdd.filter(gt50).collect())

# alternately, declaring and using lambda function on the fly
# where lambda helps iterate through each case of list
print(rdd.filter(lambda i: i > 50).collect())

# shuffling will return a differently ordered list of the same numbers each time.
# there must be a way to standardize that.

# implying two conditions in separate functions, and summing the result, using methods
print(rdd.filter(lambda x: x > 50).filter(lambda x: x < 75).sum())





























# in order to display plot within window
# plt.show()
