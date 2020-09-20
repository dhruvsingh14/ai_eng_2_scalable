############################
# Week 3.1: Spark ML Intro #
############################

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

##########################
# Importing Dataset: Hmp #
##########################
# deleting existing files from previous runs
import os
os.remove('hmp.parquet')

# importing libraries
import wget

# downloading parquet format data
url = 'https://github.com/IBM/coursera/raw/master/hmp.parquet'
wget.download(url, 'hmp.parquet')

# reading and storing dataset
df = spark.read.parquet('hmp.parquet')

# registering a corresponding query table / spark dataframe
df.createOrReplaceTempView('hmp')

print(df.show())

#############################################################
# Pulling in and tabulating iot data: MinMaxScaler Included #
#############################################################
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

# transforming our vectors for easier ingestion
indexer = StringIndexer(inputCol = "class", outputCol="classIndex")
encoder = OneHotEncoder(inputCol = "classIndex", outputCol="categoryVec")
vectorAssembler = VectorAssembler(inputCols=["x", "y", "z"],
                                    outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

# resets range, and bounds, mean and sd. for standardization
minmaxscaler = MinMaxScaler(inputCol="features", outputCol="features_minmax")

# transformers vs. estimators
# transformers simply transform a vector
# estimators additionally have a fit function attached to them
pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, normalizer, minmaxscaler])
model = pipeline.fit(df)
prediction = model.transform(df)
prediction.show()

#################################################################
# Pulling in and tabulating iot data: MinMaxScaler Not Included #
#################################################################
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

# transforming our vectors for easier ingestion
indexer = StringIndexer(inputCol = "class", outputCol="classIndex")
encoder = OneHotEncoder(inputCol = "classIndex", outputCol="categoryVec")
vectorAssembler = VectorAssembler(inputCols=["x", "y", "z"],
                                    outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

# transformers vs. estimators
# transformers simply transform a vector
# estimators additionally have a fit function attached to them
pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, normalizer])
model = pipeline.fit(df)
prediction = model.transform(df)
prediction.show()
































# in order to display plot within window
# plt.show()
