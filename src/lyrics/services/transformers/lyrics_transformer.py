from abc import abstractmethod

from pyspark.sql import DataFrame
from pyspark.ml import Transformer
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
