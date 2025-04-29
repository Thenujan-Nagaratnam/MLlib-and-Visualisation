from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import regexp_replace, trim, col
from src.lyrics.column import Column
from src.lyrics.services.transformers.lyrics_transformer import LyricsTransformer


class Cleanser(LyricsTransformer):
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        dataframe = dataframe.withColumn(
            Column.CLEAN.value, regexp_replace(trim(col(Column.VALUE.value)), r"[^\w\s]", "")
        )
        dataframe = dataframe.withColumn(
            Column.CLEAN.value, regexp_replace(col(Column.CLEAN.value), r"\s{2,}", " ")
        )
        dataframe = dataframe.drop(Column.VALUE.value)
        dataframe = dataframe.filter(col(Column.CLEAN.value).isNotNull())
        return dataframe
