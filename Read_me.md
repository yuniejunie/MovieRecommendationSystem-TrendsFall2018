#Recommender System - Recommend Movies for a New User using pyspark and AWS

Recommender sytsem 

## AWS and PySpark

### Load packages and set sqlContext
```pyspark3
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import math
import sys

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import isnan, isnull
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import functions as f

from pyspark.mllib.recommendation import ALS as rdd_als
from pyspark.sql.functions import lit
import math

sqlContext=SQLContext(sc)
```

### Read the movie and rating data on AWS S3 
We used the data from [Movielens](https://grouplens.org/datasets/movielens/). There are 2 data set we are using:
1) rating.csv, including userId, movieId, rating, timestamp
2) movies.csv, including movieId, movieName, genre 

```pyspark3
# read rating file
filename='s3://trendsmarketplacemsba2018fall/Ratings/ratings.csv'
colum1=[
    StructField("userId",LongType()),
    StructField("movieId",LongType()),
    StructField("rating",DoubleType()),
    StructField("timestamp",LongType())
]
schm1=StructType(colum1)
ratingDF=sqlContext.read.format('csv').schema(schm1).option("header",'true').load(filename)
ratingDF=ratingDF.select(ratingDF.userId, ratingDF.movieId, ratingDF.rating)
ratingDF.cache()

# read movie file
filename='s3://trendsmarketplacemsba2018fall/Movies/movies.csv'
colum=[
    StructField("movieId",LongType()),
    StructField("movieName",StringType()),
    StructField("genre",StringType())
]
schm=StructType(colum)
movieDF=sqlContext.read.format('csv').schema(schm).option("header",'true').load(filename)
```

### For each movie, calculate how many reviews it has received from users. 
```pyspark3
# caculate movie review
movie_reviews_count=ratingDF.groupBy(ratingDF.movieId).agg({"movieId":"count", "rating":"avg"})
movie_reviews_count = movie_reviews_count.select("movieId", 
 f.col("avg```(rating)").alias("Avg_Rating"),
 f.col("count(movieId)").alias("No_Reviews"))
```

### Define the function to get the new user movie recommendations based on their initial ratings on sample movies.
The function takes 4 inputs:
- inputname: the csv file which include the initial rating from a new user
- outputname: the path on AWS S3 to save the output
- ratingDF: the rating spark dataframe
- movieDF: the movie spark dataframe
- movie_reviews_count: the precaculated aggregated movie information spark dataframe
```pyspark3
def get_recommendation(inputname,outputname,ratingDF,movieDF,movie_reviews_count):    
    colum2=[
        StructField("userId",LongType()),
        StructField("movieId",LongType()),
        StructField("movieName",StringType()),
        StructField("rating",DoubleType())
    ]
    schm2=StructType(colum2)
    new_user_rating_df=sqlContext.read.format('csv').schema(schm2).option("header",'true').load(inputname)
    new_user_rating_df = new_user_rating_df.select("userId","movieId","rating")
    new_Id = [int(i.userId) for i in new_user_rating_df.select("userId").limit(1).collect()][0]
    
    #New rating form
    ratingDF_new = ratingDF.sample(False,0.2).union(new_user_rating_df)
    rating_rdd = ratingDF_new.rdd
    new_ratings_model = rdd_als.train(rating_rdd,5,10)

    ids = new_user_rating_df.select(new_user_rating_df.movieId)
    new_ids = [int(i.movieId) for i in ids.collect()]
    new_user_df =  movieDF.filter(movieDF.movieId.isin(*new_ids) == False)
    new_user_rdd=new_user_df.rdd.map(lambda x: (new_Id,x[0] ))

    new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_rdd)
    new_user_recommendations_df=new_user_recommendations_RDD.toDF()
    final = new_user_recommendations_df.join(movie_reviews_count,
     new_user_recommendations_df.product == movie_reviews_count.movieId, 
     how="left")
    final = final.filter(final.No_Reviews >= 5000)
    final = final.sort("rating", ascending=False)
    final = final.select('movieId','rating','No_Reviews','Avg_Rating').join(movieDF, "movieId", how='left').limit(10)
    final.write.csv(outputname)#s3://trendsmarketplacemsba2018fall/output/output.csv
    return final
```

### Demonstration

```pyspark3
inputname="s3://trendsmarketplacemsba2018fall/Input/new_user.csv"
outputname="s3://trendsmarketplacemsba2018fall/output/output"
get_recommendation(inputname,outputname,ratingDF,movieDF,movie_reviews_count)
```



