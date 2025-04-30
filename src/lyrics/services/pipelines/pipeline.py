# $example on$
from pyspark.ml.classification import LogisticRegression
from typing import cast
from pyspark.sql.dataframe import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Tokenizer, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer

from src.lyrics.column import Column
from src.lyrics.services.pipelines.lyrics_pipeline import LyricsPipeline
from src.lyrics.services.transformers.cleanser import Cleanser
from src.lyrics.services.transformers.label_encoder import LabelEncoder
from src.lyrics.services.transformers.stemmer import Stemmer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# $example off$
from pyspark.sql import SparkSession


class LogisticRegressionPipeline:
    def __init__(self, dataset_path, train_ratio, print_statistics, store_model_on):
        self.spark = SparkSession.builder.appName(
            "LogisticRegressionPipeline"
        ).getOrCreate()

        self.print_statistics = print_statistics
        self.dataset_path = dataset_path
        self.train_ratio = train_ratio
        self.store_model_on = store_model_on

    def read_csv(self, path) -> DataFrame:
        return self.spark.read.csv(path, header=True, inferSchema=True)

    def train_and_test(self):

        data = self.read_csv(self.dataset_path)

        train_df, test_df = data.randomSplit(
            [self.train_ratio, (1 - self.train_ratio)], seed=42
        )

        # train_df: DataFrame = train_df.select(Column.VALUE.value, Column.GENRE.value)

        label_encoder = LabelEncoder()

        cleanser = Cleanser()

        tokenizer = Tokenizer(
            inputCol=Column.CLEAN.value,
            outputCol=Column.WORDS.value,
        )

        stop_words_remover = StopWordsRemover(
            inputCol=Column.WORDS.value,
            outputCol=Column.FILTERED_WORDS.value,
        )

        stemmer = Stemmer()

        word_to_vec = Word2Vec(
            inputCol=Column.STEMMED_WORDS.value,
            outputCol=Column.FEATURES.value,
            minCount=0,
            seed=42,
        )

        lr = LogisticRegression(
            featuresCol=Column.FEATURES.value,
            labelCol=Column.LABEL.value,
            predictionCol=Column.PREDICTION.value,
            probabilityCol=Column.PROBABILITY.value,
        )

        pipeline = Pipeline(
            stages=[
                label_encoder,
                cleanser,
                tokenizer,
                stop_words_remover,
                stemmer,
                word_to_vec,
                lr,
            ]
        )

        model: LogisticRegression = pipeline.fit(train_df)

        if self.print_statistics:
            print(f"MODEL STATISTICS: {self.get_model_statistics(model)}")

        self.test(test_df, model, self.store_model_on)

        if self.store_model_on:
            model.write().overwrite().save(self.store_model_on)

        return model

    def test(
        self,
        dataframe: DataFrame,
        model: LogisticRegression = None,
        saved_model_dir_path: str = None,
    ) -> float:
        if not model:
            model = LogisticRegression.load(saved_model_dir_path)

        predictions = model.transform(dataframe)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        test_accuracy = evaluator.evaluate(predictions)
        print("Test Error = %g" % (1.0 - test_accuracy))

        return model
