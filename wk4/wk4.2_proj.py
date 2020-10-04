#####################
# Week 4.2: Project #
#####################

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

##############################
# Importing Dataset: Weather #
##############################
# delete existing files from previous runs
import os
os.remove('jfk_weather.tar.gz')

# importing libraries
import wget

# downloading tar zip file containing data
url = 'https://dax-cdn.cdn.appdomain.cloud/dax-noaa-weather-data-jfk-airport/1.1.4/noaa-weather-data-jfk-airport.tar.gz'
wget.download(url, 'jfk_weather.tar.gz')

import tarfile
# extracting tarball
my_tar = tarfile.open('jfk_weather.tar.gz')
my_tar.extractall()
my_tar.close()

# creating a dataframe, using csv
df = spark.read.option("header", "true").option("inferSchema", "true").csv('noaa-weather-data-jfk-airport/jfk_weather.csv')

# registering a corresponding query table / spark dataframe
df.createOrReplaceTempView('df')

##################
# Importing Data #
##################
import random
random.seed(42)

from pyspark.sql.functions import translate, col

df_cleaned = df \
    .withColumn("HOURLYWindSpeed", df.HOURLYWindSpeed.cast('double')) \
    .withColumn("HOURLYWindDirection", df.HOURLYWindDirection.cast('double')) \
    .withColumn("HOURLYStationPressure", translate(col("HOURLYStationPressure"), "s,", "")) \
    .withColumn("HOURLYPrecip", translate(col("HOURLYPrecip"), "s,", "")) \
    .withColumn("HOURLYRelativeHumidity", translate(col("HOURLYRelativeHumidity"), "*", "")) \
    .withColumn("HOURLYDRYBULBTEMPC", translate(col("HOURLYDRYBULBTEMPC"), "*", "")) \

df_cleaned = df_cleaned \
                    .withColumn("HOURLYStationPressure", df_cleaned.HOURLYStationPressure.cast('double')) \
                    .withColumn("HOURLYPrecip", df_cleaned.HOURLYPrecip.cast('double')) \
                    .withColumn("HOURLYRelativeHumidity", df_cleaned.HOURLYRelativeHumidity.cast('double')) \
                    .withColumn("HOURLYDRYBULBTEMPC", df_cleaned.HOURLYDRYBULBTEMPC.cast('double')) \

df_filtered = df_cleaned.filter("""
    HOURLYWindSpeed <> 0
    and HOURLYWindSpeed IS NOT NULL
    and HOURLYWindDirection IS NOT NULL
    and HOURLYStationPressure IS NOT NULL
    and HOURLYPressureTendency IS NOT NULL
    and HOURLYPrecip IS NOT NULL
    and HOURLYRelativeHumidity IS NOT NULL
    and HOURLYDRYBULBTEMPC IS NOT NULL
""")

#####################
# Building Pipeline #
#####################
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=["HOURLYWindSpeed", "HOURLYWindDirection", "HOURLYStationPressure"],
                                  outputCol="features")
df_pipeline = vectorAssembler.transform(df_filtered)
from pyspark.ml.stat import Correlation
print(Correlation.corr(df_pipeline, "features").head()[0].toArray())
# we see a corr of .25 bw windspeed and winddirection
# and of -.26 bw windspeed and stationpressure

####################
# Test-Train Split #
####################
splits = df_filtered.randomSplit([0.8, 0.2])
df_train = splits[0]
df_test = splits[1]

#####################
# Re-Using Pipeline #
#####################
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer
from pyspark.ml import Pipeline

vectorAssembler = VectorAssembler(inputCols=[
                                    "HOURLYWindDirection",
                                    "ELEVATION",
                                    "HOURLYStationPressure"],
                                  outputCol = "features")

normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

################################
# Defining Evaluation Function #
################################
def regression_metrics(prediction):
    from pyspark.ml.evaluation import RegressionEvaluator
    evaluator = RegressionEvaluator(
    labelCol="HOURLYWindSpeed", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(prediction)
    print("RMSE on test data = %g" % rmse)

##########################
# LR1: Linear Regression #
##########################
# using linear regression for baselining
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="HOURLYWindSpeed", featuresCol='features', maxIter=100, regParam=0.0, elasticNetParam=0.0)
pipeline = Pipeline(stages=[vectorAssembler, normalizer, lr])
model = pipeline.fit(df_train)

prediction = model.transform(df_test)

regression_metrics(prediction)
# rmse of 5.29 on lienar regression model

#########################################
# GBT1: Gradient Boosted Tree Regressor #
#########################################
from pyspark.ml.regression import GBTRegressor
'''
gbt = GBTRegressor(labelCol="HOURLYWindSpeed", maxIter=100)
pipeline = Pipeline(stages=[vectorAssembler, normalizer, gbt])

model = pipeline.fit(df_train)

prediction = model.transform(df_test)
regression_metrics(prediction)
# smaller rmse for gbt of 5.07
'''
# this takes wayy too long to run

####################################
# Classification: using Bucketizer #
####################################
from pyspark.ml.feature import Bucketizer, OneHotEncoder

bucketizer = Bucketizer(splits = [0, 180, float('Inf') ],
                        inputCol="HOURLYWindDirection",
                        outputCol = "HOURLYWindDirectionBucketized")

encoder = OneHotEncoder(inputCol="HOURLYWindDirectionBucketized",
                        outputCol="HOURLYWindDirectionOHE")

###################################
# Performance Evaluation Function #
###################################
def classification_metrics(prediction):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    mcEval = MulticlassClassificationEvaluator().setMetricName("accuracy").setPredictionCol("prediction").setLabelCol("HOURLYWindDirectionBucketized")
    accuracy = mcEval.evaluate(prediction)
    print("Accuracy on test data = %g" % accuracy)

############################
# LR1: Logistic Regression #
############################
# using logistic regression for baselining
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="HOURLYWindDirectionBucketized",
                        maxIter=100)
# "ELEVATION", "HOURLYStationPressure", "HOURLYPressureTendency", "HOURLYPrecip"

vectorAssembler = VectorAssembler(inputCols=[
                                    "HOURLYWindSpeed",
                                    "HOURLYDRYBULBTEMPC"],
                                  outputCol = "features")

pipeline = Pipeline(stages=[bucketizer, vectorAssembler, normalizer, lr])
model = pipeline.fit(df_train)

prediction = model.transform(df_test)

classification_metrics(prediction)
# accuracy is 0.678

######################
# RF1: Random Forest #
######################
# applying algorithms to improve performance
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="HOURLYWindDirectionBucketized", numTrees=30)

vectiorAssembler = VectorAssembler(inputCols=[
                                        "HOURLYWindSpeed",
                                        "HOURLYDRYBULBTEMPC",
                                        "ELEVATION",
                                        "HOURLYStationPressure",
                                        "HOURLYPressureTendency",
                                        "HOURLYPrecip"],
                                    outputCol="features")

pipeline = Pipeline(stages=[bucketizer, vectorAssembler, normalizer, rf])
model = pipeline.fit(df_train)
prediction = model.transform(df_test)
classification_metrics(prediction)
# model accuracy is now 0.691


###############################
# GBT2: Gradient Boosted Tree #
###############################
# applying algorithms to improve performance
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(labelCol="HOURLYWindDirectionBucketized", maxIter=100)

vectiorAssembler = VectorAssembler(inputCols=[
                                        "HOURLYWindSpeed",
                                        "HOURLYDRYBULBTEMPC",
                                        "ELEVATION",
                                        "HOURLYStationPressure",
                                        "HOURLYPressureTendency",
                                        "HOURLYPrecip"],
                                    outputCol="features")

pipeline = Pipeline(stages=[bucketizer, vectorAssembler, normalizer, gbt])
model = pipeline.fit(df_train)
prediction = model.transform(df_test)
classification_metrics(prediction)
# accuracy is now 0.688


# that concludes pyspark and machine learning as part of this specialization.


























# in order to display plot within window
# plt.show()
