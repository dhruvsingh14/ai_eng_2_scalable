############################################
# Week 2.1: Statistics and Transformations #
############################################

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

##########################################################
# Importing Dataset: Sensor Data on Household Activities #
##########################################################
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
df.createOrReplaceTempView('df')

#################################################
# Bar Chart: Counts Time Spent by Acitvity Type #
#################################################

# checking table, and schema setup
df.show()
df.printSchema()

# grouping by class to get counts
df.groupBy('class').count().show()

# plotting bar chart of category counts of time spent by activity type
import matplotlib.pyplot as plt
from pyspark.sql.functions import col

# ordering on counts
counts = df.groupBy('class').count().orderBy('count')

# removing this codepiece from original:
# display(counts)

# that's a pretty good jugaad!
df_pandas = counts.toPandas()

# sorting values descending
df_pandas = df_pandas.sort_values(by='count', ascending=False)

# Create a horizontal bar plot
df_pandas.plot(kind='barh', x='class', y='count', colormap='winter_r')
plt.show()


####################################################################
# Query: Spread Stats using Query string and SQL inbuilt Functions #
####################################################################

# quite a neat sophisticated basic query
# to tally up time spend on various tasks, w/ no inducation of the type itself.
spark.sql('''
    select
        *,
        max/min as minmaxratio -- computes min to max ratio
        from (
            select
                min(ct) as min, -- computes min val of all classes
                max(ct) as max, -- computes max val of all classes
                mean(ct) as mean, -- computes mean val b/w all classes
                stddev(ct) as stddev -- computes stddev b/w all classes

                from (
                    select
                        count(*) as ct -- counts no. rows by class
                        from df -- accesses the temporary query table
                        group by class -- aggregates over class

                )
        )
''').show()


#################################################################
# Query: Same Spread Statistics using inbuilt Functions Pyspark #
#################################################################

from pyspark.sql.functions import col, min, max, mean, stddev

df \
    .groupBy('class') \
    .count() \
    .select([
        min(col("count")).alias('min'),
        max(col("count")).alias('max'),
        mean(col("count")).alias('mean'),
        stddev(col("count")).alias('stddev')
    ]) \
    .select([
        col('*'),
        (col("max") / col("min")).alias('minmaxratio')
    ]) \
    .show()


###################################################
# Practice 1: Counts by Activity, ascending order #
###################################################
# ordering on counts
counts = df.groupBy('class').count().orderBy('count')

df_pandas = counts.toPandas()

# sorting values ascending
df_pandas = df_pandas.sort_values(by='count', ascending=True)

#################################################
# Practice 2: Plotting data by Ascending counts #
#################################################
# Create a horizontal bar plot
df_pandas.plot(kind='barh', x='class', y='count', colormap='winter_r')
plt.show()


#################
# Undersampling #
#################

from pyspark.sql.functions import min

# creates disjoint non-overlapping classes in the dataset
classes = [row[0] for row in df.select('class').distinct().collect()]

# counts class elements, limits class samples
min = df.groupBy('class').count().select(min('count')).first()[0]

# df outputted
df_balanced = None

# remember, classes are partitioned, non intersecting
for cls in classes:

    # given current class_elements
    # shuffle (using fraction = 1.0 option for sample command)
    # returns only first n samples
    df_temp = df \
        .filter("class = '"+cls+"'") \
        .sample(False, 1.0) \
        .limit(min)

    # assigning df_temp to empty df_balanced
    if df_balanced == None:
        df_balanced = df_temp

    # appending
    else:
        df_balanced=df_balanced.union(df_temp)


###################################################
# Practice 3: Checking df_balanced class_elements #
###################################################

# print(df_balanced.head())

df_balanced.groupBy('class').count().show()


























# in order to display plot within window
# plt.show()
