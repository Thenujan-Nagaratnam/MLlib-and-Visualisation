from abc import abstractmethod
from typing import Optional, cast

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import lit
from pyspark.ml import PipelineModel
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from src.lyrics.column import Column
from src.lyrics.genre import label_to_genre_map


class LyricsPipeline:
    def __init__(self) -> None:
        print("STARTING SPARK SESSION")
        self.spark = SparkSession.builder\
            .appName("LyricsClassifierPipeline")\
            .config("spark.driver.memory", "3G")\
            .config("spark.executor.memory", "3G")\
            .config("spark.executor.cores", "3")\
            .config("spark.python.worker.memory", "3G") \
            .config("spark.driver.port", "4040")\
            .getOrCreate()

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

        model: CrossValidatorModel = self.train(train_df, print_statistics)
        test_accuracy: float = self.test(test_df, model)

        if print_statistics:
            print(f"CROSS VALIDATOR MODEL AVERAGE METRICS: {model.avgMetrics}")
            print(f"TEST ACCURACY: {test_accuracy}")

        if store_model_on:
            model.write().overwrite().save(store_model_on)

        return model

    @abstractmethod
    def train(self, dataframe: DataFrame, print_statistics: bool) -> CrossValidatorModel:
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
        unknown_lyrics_df = self.spark.createDataFrame([(unknown_lyrics,)], [Column.VALUE.value])
        unknown_lyrics_df = unknown_lyrics_df.withColumn(Column.GENRE.value, lit("UNKNOWN"))

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
