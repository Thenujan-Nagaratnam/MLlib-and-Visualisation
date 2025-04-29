from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from src.lyrics.column import Column
from src.lyrics.genre import genre_to_label_map
from src.lyrics.services.transformers.lyrics_transformer import LyricsTransformer


class LabelEncoder(LyricsTransformer):
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        genre_to_label_udf = udf(
            lambda genre: genre_to_label_map.get(genre, genre_to_label_map["unknown"]), IntegerType()
        )
        dataframe = dataframe.withColumn(Column.LABEL.value, genre_to_label_udf(col(Column.GENRE.value)))
        dataframe = dataframe.drop(Column.GENRE.value)
        return dataframe
