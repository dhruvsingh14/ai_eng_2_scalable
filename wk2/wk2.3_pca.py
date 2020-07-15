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
df.createOrReplaceTempView('washing')

print(df.show())

#############################
#  #
#############################

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

print(df.count() - result.count())

# importing libraries
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler



















# in order to display plot within window
# plt.show()
