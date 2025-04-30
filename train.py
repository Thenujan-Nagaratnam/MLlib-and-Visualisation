import os
from pyspark.ml.tuning import CrossValidatorModel
from src.lyrics.services.pipelines.lr_pipeline import LogisticRegressionPipeline

DATASET_PATH = os.path.abspath("data/mendeley-music-dataset.csv")

MODEL_DIR_PATH = os.path.abspath("model/")

pipeline: LogisticRegressionPipeline
model: CrossValidatorModel


def on_startup():
    global pipeline
    global model

    if not (
        os.path.exists(MODEL_DIR_PATH)
        and os.path.isdir(MODEL_DIR_PATH)
        and len(os.listdir(MODEL_DIR_PATH)) > 0
    ):
        os.makedirs(MODEL_DIR_PATH, exist_ok=True)

        pipeline = LogisticRegressionPipeline()

        print("PLEASE WAIT UNTIL YOU SEE => INFO: Application startup complete.")

        model = pipeline.train_and_test(
            dataset_path=DATASET_PATH,
            train_ratio=0.8,
            store_model_on=MODEL_DIR_PATH,
            print_statistics=True,
        )
    else:
        pipeline = LogisticRegressionPipeline()

        print("PLEASE WAIT UNTIL YOU SEE => INFO: Application startup complete.")

        model = CrossValidatorModel.load(MODEL_DIR_PATH)


print("PLEASE WAIT UNTIL YOU SEE => INFO: Application startup complete.")

on_startup()
