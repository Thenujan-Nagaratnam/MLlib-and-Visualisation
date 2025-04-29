from nltk.stem import SnowballStemmer
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType
from src.lyrics.column import Column
from src.lyrics.services.transformers.lyrics_transformer import LyricsTransformer


class Stemmer(LyricsTransformer):
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        stemmer = SnowballStemmer("english")
        stem_udf = udf(lambda words: [stemmer.stem(word) for word in words], ArrayType(StringType()))
        dataframe = dataframe.withColumn(Column.STEMMED_WORDS.value, stem_udf(col(Column.FILTERED_WORDS.value)))
        dataframe = dataframe.select(Column.STEMMED_WORDS.value, Column.LABEL.value)
        return dataframe
