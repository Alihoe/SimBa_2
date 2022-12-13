from nerd import nerd_client
from nltk import word_tokenize
from nltk.corpus import wordnet as wn


def get_named_entities_of_sentence(sentence, entity_fisher):
    try:
        entities = entity_fisher.disambiguate_text(sentence, language='en')[0]['entities']
    except:
        print('Error occured for: ' + sentence)
        entities = []
    entity_list = []
    for entity in entities:
        if 'wikipediaExternalRef' in entity:
            wikipedia_id = entity['wikipediaExternalRef']
            entity_list.append(wikipedia_id)
        if 'wikidataId' in entity:
            wikidata_id = entity['wikidataId']
            entity_list.append(wikidata_id)
    return entity_list


def get_sequence_entities(sequence_dictionary, ref_feature):
    entity_dict = {}
    if ref_feature == "ne_similarity":
        entity_fisher = nerd_client.NerdClient()
        for id, text in sequence_dictionary.items():
            entity_dict[id] = get_named_entities_of_sentence(text, entity_fisher)
    elif ref_feature == "synonym_similarity":
        for id, text in sequence_dictionary.items():
            pp_text = set(word_tokenize(text))
            synsets_text = []
            for word in pp_text:
                synsets = wn.synsets(word)
                if synsets:
                    for synset in synsets:
                        synset_name = synset.name()
                        try:
                            index = synset_name.index('.')
                        except ValueError:
                            index = len(synset_name)
                        synset_name = synset_name[:index]
                        if synset_name not in synsets_text and len(synset_name):
                            synsets_text.append(synset_name)
            entity_dict[id] = synsets_text
    return entity_dict


def ne_sim(query_text, target_text):
    entity_fisher = nerd_client.NerdClient()
    query_nes = set(get_named_entities_of_sentence(query_text, entity_fisher))
    target_nes = set(get_named_entities_of_sentence(target_text, entity_fisher))
    len_query_entities = len(query_nes)
    len_target_entities = len(target_nes)
    len_intersection = len(query_nes.intersection(target_nes))
    if len_query_entities + len_target_entities > 0:
        return (100 / (len_query_entities + len_target_entities)) * len_intersection
    else:
        return 0


def get_text_synonyms(text):
    pp_text = set(word_tokenize(text))
    synsets_text = []
    for word in pp_text:
        synsets = wn.synsets(word)
        if synsets:
            for synset in synsets:
                synset_name = synset.name()
                try:
                    index = synset_name.index('.')
                except ValueError:
                    index = len(synset_name)
                synset_name = synset_name[:index]
                if synset_name not in synsets_text and len(synset_name):
                    synsets_text.append(synset_name)
    return synsets_text


def get_synonym_ratio(query_text, target_text):
    query_syns = set(get_text_synonyms(query_text))
    target_syns = set(get_text_synonyms(target_text))
    len_query_entities = len(query_syns)
    len_target_entities = len(target_syns)
    len_intersection = len(query_syns.intersection(target_syns))
    return (100 / (len_query_entities + len_target_entities)) * len_intersection

