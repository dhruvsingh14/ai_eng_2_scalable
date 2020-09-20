#################################
# Week 3.2: Spark ML Clustering #
#################################

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

print(df.count(), len(df.columns))

######################################
# Pulling in and tabulating iot data #
######################################
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

# transforming our vectors for easier ingestion
indexer = StringIndexer(inputCol = "class", outputCol="classIndex")
encoder = OneHotEncoder(inputCol = "classIndex", outputCol="categoryVec")
vectorAssembler = VectorAssembler(inputCols=["x", "y", "z"],
                                    outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, normalizer])
model = pipeline.fit(df)
prediction = model.transform(df)
prediction.show()

##############################
# K Means Specific Libraries #
##############################
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# k = 14
kmeans = KMeans(featuresCol="features").setK(14).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler, kmeans])
model = pipeline.fit(df)
predictions = model.transform(df)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)

print("Silhouette with square euclidean distance = " + str(silhouette))

# for k frmo 2 to 13
'''
for x in range(2,14):
        kmeans = KMeans(featuresCol="features").setK(x).setSeed(1)
        pipeline = Pipeline(stages=[vectorAssembler, kmeans])
        model = pipeline.fit(df)
        predictions = model.transform(df)

        evaluator = ClusteringEvaluator()

        silhouette = evaluator.evaluate(predictions)

        print("Silhouette with square euclidean distance = " + str(silhouette))
'''
# k = 14 is still the clear optimizer, that minimizes distances

##################################################
# Extending Model to Include Normalized Features #
##################################################
# doesn't let me input features_norm as featuresCol
kmeans = KMeans(featuresCol="features").setK(14).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler, kmeans, normalizer])
model = pipeline.fit(df)
predictions = model.transform(df)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)

print("Silhouette with square euclidean distance = " + str(silhouette))

##########################
# Inflating Dataset x 10 #
##########################
from pyspark.sql.functions import col
df_denormalized = df.select([col('*'), (col('x')*10)]).drop('x').withColumnRenamed('(x * 10)', 'x')


kmeans = KMeans(featuresCol = "features").setK(14).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler, kmeans])
model = pipeline.fit(df_denormalized)
predictions = model.transform(df_denormalized)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

##################################
# Implementing Gaussian Matrices #
##################################

# Gaussian Matrixes in the service of Clustering
from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture(featuresCol = "features").setK(14).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler, kmeans])

model = pipeline.fit(df)

predictions = model.transform(df)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with square euclidean distance = " + str(silhouette))







































# in order to display plot within window
# plt.show()
