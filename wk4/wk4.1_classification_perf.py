########################################
# Week 4.1: Classification Performance #
########################################

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

######################################
# Pulling in and tabulating iot data #
######################################
# test train split, supervised learning
splits = df.randomSplit([0.8, 0.2])
df_train = splits[0]
df_test = splits[1]

######################################
# Pulling in and tabulating iot data #
######################################
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer

# transforming our vectors for easier ingestion
indexer = StringIndexer(inputCol = "class", outputCol="label")

vectorAssembler = VectorAssembler(inputCols=["x", "y", "z"],
                                    outputCol="features")

normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

############################
# Classification Libraries #
############################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
pipeline = Pipeline(stages=[indexer, vectorAssembler, normalizer,lr])
model = pipeline.fit(df_train)
prediction = model.transform(df_test)

prediction.printSchema()


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
classifier = MulticlassClassificationEvaluator().setMetricName("accuracy").evaluate(prediction)

print("Classifier with prediction accuracy = " + str(classifier))
# 20% accuracy, not bad

######################################################
# Using Random Forest Classifier to Improve Accuracy #
######################################################
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# fitting metadata labels to index the data
labelIndexer = StringIndexer(inputCol="source", outputCol="indexedLabel").fit(df)

# df.show(n=5)

# features.show(n=5)

# to identify categorical features
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4)

#.fit(df)


# resplitting the data
(trainingData, testData) = df.randomSplit([0.8, 0.2])


# training a random forest model
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)


# converting indices to labels
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                labels=labelIndexer.labels)


pipeline = Pipeline(stages=[labelIndexer, vectorAssembler, featureIndexer, rf, labelConverter])

model = pipeline.fit(trainingData)

predictions = model.transform(testData)

# model evaluation, last part.































# in order to display plot within window
# plt.show()
