########################
# Week 1.3: Dataframes #
########################

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

###############################
# Creating an Empty Dataframe #
###############################

from pyspark.sql import Row

df = spark.createDataFrame([Row(id=1, value='value1'), Row(id=2, value='value2')])

# checking dataframe with hardcoded values
df.show()

# checking some basic df specs
df.printSchema()

###################################
# Planting Df into a query schema #
###################################

df.createOrReplaceTempView('df_view')

# querying our table
df_result = spark.sql('select value from df_view where id=2')

# printing results
df_result.show()

# fetching results as a string instead
df_result.first().value

###########################
# Row Count of Dataframes #
###########################
print(df.count())






























# in order to display plot within window
# plt.show()
