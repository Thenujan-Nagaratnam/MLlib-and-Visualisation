from typing import cast, List

from pyspark.sql.dataframe import DataFrame
from pyspark.ml import Pipeline, PipelineModel, Transformer
from pyspark.ml.feature import StopWordsRemover, Tokenizer, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import (
    CrossValidator,
    CrossValidatorModel,
    ParamGridBuilder,
    # TrainValidationSplit,
    # TrainValidationSplitModel,
)
from src.lyrics.column import Column
from src.lyrics.services.pipelines.lyrics_pipeline import LyricsPipeline
from src.lyrics.services.transformers.cleanser import Cleanser
from src.lyrics.services.transformers.label_encoder import LabelEncoder
from src.lyrics.services.transformers.stemmer import Stemmer


class LogisticRegressionPipeline(LyricsPipeline):
    def train(
        self,
        dataframe: DataFrame,
        print_statistics: bool = False,
    ) -> CrossValidatorModel:
        dataframe: DataFrame = dataframe.select(Column.VALUE.value, Column.GENRE.value)

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
            print(f"MODEL STATISTICS: {self.get_model_statistics(cross_validator_model)}")

        return cross_validator_model

    @staticmethod
    def get_model_statistics(model: CrossValidatorModel) -> dict:
        model_statistics = LyricsPipeline.get_model_basic_statistics(model)

        best_model: PipelineModel = cast(PipelineModel, model.bestModel)
        stages: List[Transformer] = best_model.stages

        model_statistics["RegParam"] = cast(LogisticRegression, stages[-1]).getRegParam()
        model_statistics["MaxIter"] = cast(LogisticRegression, stages[-1]).getMaxIter()
        model_statistics["VectorSize"] = cast(Word2Vec, stages[-2]).getVectorSize()

        return model_statistics
