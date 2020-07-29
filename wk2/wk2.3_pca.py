#################
# Week 2.3: PCA #
#################

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

#############################
# Importing Dataset: Washer #
#############################
# deleting existing files from previous runs
import os
os.remove('washing.parquet')

# importing libraries
import wget

# downloading parquet format data
url = 'https://github.com/IBM/coursera/blob/master/coursera_ds/washing.parquet?raw=true'
wget.download(url, 'washing.parquet')

# reading and storing dataset
df = spark.read.parquet('washing.parquet')

# registering a corresponding query table / spark dataframe
df.createOrReplaceTempView('washing')

print(df.show())

################################################
# Selecting all the relevant fields of inquiry #
################################################
# restricting our query to max and mins of each field
result = spark.sql("""
SELECT * from (
    SELECT
    min(temperature) over w as min_temperature,
    max(temperature) over w as max_temperature,
    min(voltage) over w as min_voltage,
    max(voltage) over w as max_voltage,
    min(flowrate) over w as min_flowrate,
    max(flowrate) over w as max_flowrate,
    min(frequency) over w as min_frequency,
    max(frequency) over w as max_frequency,
    min(hardness) over w as min_hardness,
    max(hardness) over w as max_hardness,
    min(speed) over w as min_speed,
    max(speed) over w as max_speed
    FROM washing
    WINDOW w AS (ORDER BY ts ROWS BETWEEN CURRENT ROW AND 10 FOLLOWING)
)
WHERE min_temperature is not null
AND max_temperature is not null
AND min_voltage is not null
AND max_voltage is not null
AND min_flowrate is not null
AND max_flowrate is not null
AND min_frequency is not null
AND max_frequency is not null
AND min_hardness is not null
AND min_speed is not null
AND max_speed is not null
""")

# you shouldn't see any nulls in results dataframe since it excludes nulls
# from its calculation
print(result.show())

# checking for null rows:
# by subtracting count of total rows - count with nulls excluded
print(df.count() - result.count())

####################
# SparkML Pipeline #
####################
# importing libraries
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# compiling all features into one column
assembler = VectorAssembler(inputCols=result.columns, outputCol="features")

# transforming single column
features = assembler.transform(result)

# checking what the combined column looks like
print(features.rdd.map(lambda r : r.features).take(10))

# declaring pca object, with 3 clusters, and fitting it to single column
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(features)

# transforming single column - using pca object
result_pca = model.transform(features).select("pcaFeatures")
result_pca.show(truncate=False)
# of the 3 columns, the middle column has a different sign from our example
# may still plot correctly, only along the other side of the axis.

# printing row count - SAME
print(result_pca.count())

# sampling columns
rdd = result_pca.rdd.sample(False,0.8)

# drawing into lists, and preparing to plot
x = rdd.map(lambda a: a.pcaFeatures).map(lambda a : a[0]).collect()
y = rdd.map(lambda a: a.pcaFeatures).map(lambda a : a[1]).collect()
z = rdd.map(lambda a: a.pcaFeatures).map(lambda a : a[2]).collect()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection = '3d')

ax.scatter(x,y,z, c='r', marker = 'o')

ax.set_xlabel('dimension1')
ax.set_ylabel('dimension2')
ax.set_zlabel('dimension3')

plt.show()


























# in order to display plot within window
# plt.show()
