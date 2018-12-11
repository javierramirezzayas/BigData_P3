from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession
import csv
import re

# file = "/tmp/hive_output/000000_0"
file = "/home/jramirez/hive_output_merged"


def processTweet(tweet):
    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # trim
    tweet = tweet.strip('\'"')
    return tweet


# Define spark
conf = SparkConf().setAppName("Sentiment_Analysis")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

data = sc.textFile(file)

r = data.mapPartitions(lambda x: csv.reader(x, delimiter='\t'))
r = r.map(lambda x: processTweet(str(x)))

r = r.map(lambda x: Row(sentence=x))

df = spark.createDataFrame(r)

df.write.csv('spark_tweets')