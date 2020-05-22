##############################
# Week 1.1: Working with RDD #
##############################


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


# initializing a distributed cluster
rdd = sc.parallelize(range(100))

# running a count process
print(rdd.count())


# running a sum process
print(rdd.sum())





































# in order to display plot within window
# plt.show()
