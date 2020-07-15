######################
# Week 2.2: Plotting #
######################

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
df.createOrReplaceTempView('df')

#######################
# Checking row counts #
#######################
print(df.count())


############################
# Assigning to a Dataframe #
############################
# creating an apache spark sql dataframe
df.createOrReplaceTempView("washing")

# querying that apache spark dataframe
spark.sql("SELECT * FROM washing").show()

#####################
# Preparing Boxplot #
#####################

# checking first few rows of voltage data
result = spark.sql("select voltage from washing where voltage is not null")
result_array = result.rdd.map(lambda row : row.voltage).sample(False, 0.1).collect()

# printing the elements
print(result_array[:15])

# plotting boxplot of voltage data
import matplotlib.pyplot as plt
plt.boxplot(result_array)
plt.show()

###################################
# Lineplot: For Finer Granularity #
###################################
result = spark.sql("select voltage, ts from washing where voltage is not null order by ts asc")
result_rdd = result.rdd.sample(False, 0.1).map(lambda row : (row.ts,row.voltage))

# declaring an array for time in seconds, and voltage
result_array_ts = result_rdd.map(lambda ts_voltage: ts_voltage[0]).collect()
result_array_voltage = result_rdd.map(lambda ts_voltage: ts_voltage[1]).collect()

# printing elements of voltage and time in seconds
print(result_array_ts[:15])
print(result_array_voltage[:15])

# plotting collected data
plt.plot(result_array_ts, result_array_voltage)
plt.xlabel("time")
plt.ylabel("voltage")
plt.show()

spark.sql("select min(ts), max(ts) from washing").show()

# Line Graph 2: subset

# writing a sql query to focus in on high noise activity
result = spark.sql(
"""
select voltage, ts from washing
    where voltage is not null and
    ts > 1547808720911 and
    ts <= 1547810064867+3600000
    order by ts asc
""")
result_rdd = result.rdd.map(lambda row : (row.ts, row.voltage))

# declaring two arrays for time and voltage
result_array_ts = result_rdd.map(lambda ts_voltage: ts_voltage[0]).collect()
result_array_voltage = result_rdd.map(lambda ts_voltage: ts_voltage[1]).collect()

# plotting linegraph of subset
plt.plot(result_array_ts, result_array_voltage)
plt.xlabel("time")
plt.ylabel("voltage")
plt.show()


################################################
# 3d Scatter: For Multiple Features Comparison #
################################################
# writing a sql query to plot features hardness, temperature, and flowrate
result_df = spark.sql(
"""
select hardness, temperature, flowrate from washing
    where hardness is not null and
    temperature is not null and
    flowrate is not null
""")
result_rdd = result_df.rdd.sample(False, 0.1).map(lambda row : (row.hardness,
                                                row.temperature, row.flowrate))

# declaring three arrays for hardness, temprrature, and flowrate
result_array_hardness = result_rdd.map(lambda hardness_temperature_flowrate: hardness_temperature_flowrate[0]).collect()
result_array_temperature = result_rdd.map(lambda hardness_temperature_flowrate: hardness_temperature_flowrate[1]).collect()
result_array_flowrate = result_rdd.map(lambda hardness_temperature_flowrate: hardness_temperature_flowrate[2]).collect()

# importing 3d library from matplotlib
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plotting 3d scatter plot
ax.scatter(result_array_hardness, result_array_temperature, result_array_flowrate,
            c='r', marker='o')

ax.set_xlabel('hardness')
ax.set_ylabel('temperature')
ax.set_zlabel('flowrate')

plt.show()

# plots scatter, but for some reason, my axes are clipped relative to theirs

#######################################
# Histogram: Hardness data drill down #
#######################################
plt.hist(result_array_hardness)
plt.show()






















# in order to display plot within window
# plt.show()
