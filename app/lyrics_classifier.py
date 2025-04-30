import os
from enum import Enum
from abc import abstractmethod
from typing import cast, List, Optional
from nltk.stem import SnowballStemmer

from pyspark.sql.types import IntegerType
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import regexp_replace, trim, col
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

from pyspark.ml import Pipeline, PipelineModel, Transformer
from pyspark.ml.feature import StopWordsRemover, Tokenizer, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import (
    CrossValidator,
    CrossValidatorModel,
    ParamGridBuilder,
)
from pyspark.ml.util import (
    DefaultParamsReader,
    DefaultParamsWriter,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
)


class LyricsTransformer(Transformer, MLReadable, MLWritable):
    @abstractmethod
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        pass

    def write(self) -> MLWriter:
        return DefaultParamsWriter(self)

    @classmethod
    def read(cls) -> MLReader:
        return DefaultParamsReader(cls)


class LabelEncoder(LyricsTransformer):
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        genre_to_label_udf = udf(
            lambda genre: genre_to_label_map.get(genre, genre_to_label_map["unknown"]),
            IntegerType(),
        )
        dataframe = dataframe.withColumn(
            Column.LABEL.value, genre_to_label_udf(col(Column.GENRE.value))
        )
        dataframe = dataframe.drop(Column.GENRE.value)
        return dataframe


class Cleanser(LyricsTransformer):
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        dataframe = dataframe.withColumn(
            Column.CLEAN.value,
            regexp_replace(trim(col(Column.VALUE.value)), r"[^\w\s]", ""),
        )
        dataframe = dataframe.withColumn(
            Column.CLEAN.value, regexp_replace(col(Column.CLEAN.value), r"\s{2,}", " ")
        )
        dataframe = dataframe.drop(Column.VALUE.value)
        dataframe = dataframe.filter(col(Column.CLEAN.value).isNotNull())
        return dataframe


class Stemmer(LyricsTransformer):
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        stemmer = SnowballStemmer("english")
        stem_udf = udf(
            lambda words: [stemmer.stem(word) for word in words],
            ArrayType(StringType()),
        )
        dataframe = dataframe.withColumn(
            Column.STEMMED_WORDS.value, stem_udf(col(Column.FILTERED_WORDS.value))
        )
        dataframe = dataframe.select(Column.STEMMED_WORDS.value, Column.LABEL.value)
        return dataframe


class Column(Enum):
    VALUE = "lyrics"
    GENRE = "genre"
    LABEL = "label"
    CLEAN = "cleaned_lyrics"
    WORDS = "tokenized_lyrics"
    FILTERED_WORDS = "stop_words_removed_lyrics"
    STEMMED_WORDS = "stemmed_lyrics"
    FEATURES = "features"
    PREDICTION = "prediction"
    PROBABILITY = "probability"


genre_to_label_map = {
    "pop": 0,
    "country": 1,
    "blues": 2,
    "rock": 3,
    "jazz": 4,
    "reggae": 5,
    "hip hop": 6,
    "retro": 7,
    "unknown": 8,
}

label_to_genre_map = {
    0: "pop",
    1: "country",
    2: "blues",
    3: "rock",
    4: "jazz",
    5: "reggae",
    6: "hip hop",
    7: "retro",
    8: "unknown",
}


class LyricsPipeline:
    def __init__(self) -> None:
        print("STARTING SPARK SESSION")
        self.spark = (
            SparkSession.builder.appName("LyricsClassifierPipeline")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .config("spark.network.timeout", "600s")
            .config("spark.executor.heartbeatInterval", "60s")
            .getOrCreate()
        )

        self.spark.sparkContext.setLogLevel("ERROR")

    def stop(self) -> None:
        print("STOPPING SPARK SESSION")
        self.spark.stop()

    def read_csv(self, path) -> DataFrame:
        return self.spark.read.csv(path, header=True, inferSchema=True)

    def train_and_test(
        self,
        dataset_path: str,
        train_ratio: float,
        store_model_on: Optional[str] = None,
        print_statistics: bool = False,
    ) -> CrossValidatorModel:
        data = self.read_csv(dataset_path)
        train_df, test_df = data.randomSplit([train_ratio, (1 - train_ratio)], seed=42)

        # print("DATAFRAME INFO:")
        # print(f"Number of rows: {data.count()}")
        # print(f"Number of columns: {len(data.columns)}")
        # print("Columns and their data types:")
        # data.printSchema()

        model: CrossValidatorModel = self.train(train_df, print_statistics)
        test_accuracy: float = self.test(test_df, model)

        if print_statistics:
            print(f"CROSS VALIDATOR MODEL AVERAGE METRICS: {model.avgMetrics}")
            print(f"TEST ACCURACY: {test_accuracy}")

        if store_model_on:
            model.write().overwrite().save(store_model_on)

        return model

    @abstractmethod
    def train(
        self, dataframe: DataFrame, print_statistics: bool
    ) -> CrossValidatorModel:
        pass

    def test(
        self,
        dataframe: DataFrame,
        model: Optional[CrossValidatorModel] = None,
        saved_model_dir_path: Optional[str] = None,
    ) -> float:
        if not model:
            model = CrossValidatorModel.load(saved_model_dir_path)

        best_model: PipelineModel = cast(PipelineModel, model.bestModel)

        predictions = best_model.transform(dataframe)

        evaluator = MulticlassClassificationEvaluator(
            predictionCol=Column.PREDICTION.value,
            labelCol=Column.LABEL.value,
            metricName="accuracy",
        )

        accuracy = evaluator.evaluate(predictions)

        return accuracy

    def predict_one(
        self,
        unknown_lyrics: str,
        threshold: float,
        model: Optional[CrossValidatorModel] = None,
        saved_model_dir_path: Optional[str] = None,
    ):
        unknown_lyrics_df = self.spark.createDataFrame(
            [(unknown_lyrics,)], [Column.VALUE.value]
        )
        unknown_lyrics_df = unknown_lyrics_df.withColumn(
            Column.GENRE.value, lit("UNKNOWN")
        )

        if not model:
            model = CrossValidatorModel.load(saved_model_dir_path)

        best_model: PipelineModel = cast(PipelineModel, model.bestModel)

        predictions_df = best_model.transform(unknown_lyrics_df)
        prediction_row = predictions_df.first()

        prediction = prediction_row[Column.PREDICTION.value]
        prediction = label_to_genre_map[prediction]

        if Column.PROBABILITY.value in predictions_df.columns:
            probabilities = prediction_row[Column.PROBABILITY.value]
            probabilities = dict(zip(label_to_genre_map.values(), probabilities))

            if probabilities[prediction] < threshold:
                prediction = "UNKNOWN"

            return prediction, probabilities

        return prediction, {}

    @staticmethod
    def get_model_basic_statistics(model: CrossValidatorModel) -> dict:
        model_statistics = dict()
        model.avgMetrics.sort()
        model_statistics["Best model metrics"] = model.avgMetrics[-1]
        return model_statistics


class LogisticRegressionPipeline(LyricsPipeline):
    def train(
        self,
        dataframe: DataFrame,
        print_statistics: bool = False,
    ) -> CrossValidatorModel:
        dataframe: DataFrame = dataframe.select(Column.VALUE.value, Column.GENRE.value)

        # dataframe.printSchema()

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

        param_grid_builder = ParamGridBuilder()
        param_grid_builder.addGrid(word_to_vec.vectorSize, [500])
        param_grid_builder.addGrid(lr.regParam, [0.01])
        param_grid_builder.addGrid(lr.maxIter, [100])
        param_grid = param_grid_builder.build()

        evaluator = MulticlassClassificationEvaluator(
            predictionCol=Column.PREDICTION.value,
            labelCol=Column.LABEL.value,
            metricName="accuracy",
        )

        cross_validator = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=5,
            seed=42,
        )

        cross_validator_model: CrossValidatorModel = cross_validator.fit(dataframe)

        # tv_split = TrainValidationSplit(
        #     estimator=pipeline,
        #     estimatorParamMaps=param_grid,
        #     evaluator=MulticlassClassificationEvaluator(),
        #     trainRatio=0.8,
        #     seed=42,
        # )

        # tv_split_model: TrainValidationSplitModel = tv_split.fit(dataframe)

        if print_statistics:
            print(
                f"MODEL STATISTICS: {self.get_model_statistics(cross_validator_model)}"
            )

        return cross_validator_model

    @staticmethod
    def get_model_statistics(model: CrossValidatorModel) -> dict:
        model_statistics = LyricsPipeline.get_model_basic_statistics(model)

        best_model: PipelineModel = cast(PipelineModel, model.bestModel)
        stages: List[Transformer] = best_model.stages

        model_statistics["RegParam"] = cast(
            LogisticRegression, stages[-1]
        ).getRegParam()
        model_statistics["MaxIter"] = cast(LogisticRegression, stages[-1]).getMaxIter()
        model_statistics["VectorSize"] = cast(Word2Vec, stages[-2]).getVectorSize()

        return model_statistics
