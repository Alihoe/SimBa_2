from difflib import SequenceMatcher
from nltk import word_tokenize
from Levenshtein import ratio


def match_sequences(query_text, target_text):
    return (SequenceMatcher(a=query_text, b=target_text).ratio())*100


def levenshtein_sim(query_text, target_text):
    return ratio(query_text, target_text)*100


def jac_sim(query_text, target_text):
    a = set(word_tokenize(query_text))
    b = set(word_tokenize(target_text))
    return (float(len(a.intersection(b))) / len(a.union(b)))*100


def get_string_similarity(query, target, string_feature):
    if string_feature == "jaccard_similarity":
        return jac_sim(query, target)
    elif string_feature == "levenshtein":
        return levenshtein_sim(query, target)
    elif string_feature == "sequence_matching":
        return match_sequences(query, target)



