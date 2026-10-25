import findspark
findspark.init()

import pyspark
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan, when, count, col
import pandas as pd

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# -----------------------------
# 1. START SPARK
# -----------------------------

sc = pyspark.SparkContext(
    appName="myAppName2"
).getOrCreate()

spark = SparkSession(sc)

print("Spark Started")


# -----------------------------
# 2. CREATE CANCELLATION SCHEMA
# -----------------------------

cancel_schema = StructType([
    StructField("date", StringType(), True),
    StructField("flight_number", StringType(), True),
    StructField("flight_identifier", StringType(), True),
    StructField("cancel", StringType(), True),
    StructField("reason", StringType(), True)
])


# -----------------------------
# 3. LOAD CANCELLATION DATA
# -----------------------------

cancel_df = spark.read.format("csv") \
    .option("mode", "DROPMALFORMED") \
    .load(
        "flight-cancellation.csv",
        header=True,
        schema=cancel_schema
    )

print("Cancellation Data:")
cancel_df.show(5)


# -----------------------------
# 4. REMOVE UNNECESSARY COLUMN
# -----------------------------

cancel_df = cancel_df.drop("reason")


# -----------------------------
# 5. REMOVE NULL VALUES
# -----------------------------

cancel_df = cancel_df.dropna()


# -----------------------------
# 6. TAKE SAMPLE
# -----------------------------

cancel_df = cancel_df.sample(0.1)


# -----------------------------
# 7. CHECK SCHEMA
# -----------------------------

print("Cancellation Schema:")
cancel_df.printSchema()


# -----------------------------
# 8. COUNT DATA
# -----------------------------

print("Total Records:")
print(cancel_df.count())


# -----------------------------
# 9. CATEGORICAL COLUMNS
# -----------------------------

categoricalColumns = [
    "date",
    "flight_number",
    "flight_identifier"
]


# -----------------------------
# 10. STRING INDEX + ENCODING
# -----------------------------

stages = []

for categoricalCol in categoricalColumns:

    stringIndexer = StringIndexer(
        inputCol=categoricalCol,
        outputCol=categoricalCol + "Index"
    )

    encoder = OneHotEncoder(
        inputCols=[stringIndexer.getOutputCol()],
        outputCols=[categoricalCol + "classVec"]
    )

    stages += [
        stringIndexer,
        encoder
    ]


# -----------------------------
# 11. LABEL
# -----------------------------

label_stringIdx = StringIndexer(
    inputCol="cancel",
    outputCol="label"
)

stages += [label_stringIdx]


# -----------------------------
# 12. VECTOR ASSEMBLER
# -----------------------------

assemblerInputs = [
    c + "classVec"
    for c in categoricalColumns
]

assembler = VectorAssembler(
    inputCols=assemblerInputs,
    outputCol="features"
)

stages += [assembler]


# -----------------------------
# 13. COLUMNS
# -----------------------------

cols = [
    "date",
    "flight_number",
    "flight_identifier",
    "cancel"
]


# -----------------------------
# 14. PIPELINE
# -----------------------------

pipeline = Pipeline(
    stages=stages
)


# -----------------------------
# 15. TRANSFORM DATA
# -----------------------------

pipelineModel = pipeline.fit(cancel_df)

cancel_df = pipelineModel.transform(cancel_df)


# -----------------------------
# 16. SELECT FINAL COLUMNS
# -----------------------------

selectedCols = [
    "label",
    "features"
] + cols

cancel_df = cancel_df.select(selectedCols)


# -----------------------------
# 17. SHOW DATA
# -----------------------------

print("Prepared ML Data:")

cancel_df.show(5)


# -----------------------------
# 18. TRAIN TEST SPLIT
# -----------------------------

train, test = cancel_df.randomSplit(
    [0.7, 0.3],
    seed=2018
)

print("Training Data:")
print(train.count())

print("Testing Data:")
print(test.count())


# -----------------------------
# 19. LOGISTIC REGRESSION
# -----------------------------

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label"
)

lrModel = lr.fit(train)

prediction = lrModel.transform(test)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(prediction)

print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100)

# -----------------------------
# 20. SHOW PREDICTION
# -----------------------------

print("Logistic Regression Prediction:")

prediction.select(
    "label",
    "prediction"
).show(10)


print("Project Completed")


# -----------------------------
# 21. STOP SPARK
# -----------------------------

spark.stop()