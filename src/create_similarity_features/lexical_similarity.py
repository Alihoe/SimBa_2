from nltk import word_tokenize
from nltk.corpus import stopwords

characters = ["",'']


def get_lexical_entities(sequence_dictionary, lex_feature):
    entity_dict = {}
    if lex_feature == "similar_words_ratio" or lex_feature == "similar_words_ratio_length":
        for id, text in sequence_dictionary.items():
            entity_dict[id] = tokenize_and_filter_out_stop_words(text)
    return entity_dict


def tokenize_and_filter_out_stop_words(sequence):
    stop_words = set(stopwords.words('english'))
    return [w for w in tokenize(sequence) if not w.lower() in stop_words and not w in characters and len(w) > 1]


def tokenize(sequence):
    return word_tokenize(sequence)



