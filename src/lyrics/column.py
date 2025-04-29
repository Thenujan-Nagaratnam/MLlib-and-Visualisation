from enum import Enum


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
