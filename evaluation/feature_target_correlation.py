import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from evaluation import DATA_PATH
from evaluation.correlation_analysis import analyse_feature_correlation
from evaluation.utils import get_sim_score
from src.create_similarity_features.lexical_similarity import get_lexical_entities
from src.create_similarity_features.referential_similarity import get_sequence_entities
from src.create_similarity_features.sentence_encoder import encode_queries, encode_targets
from src.create_similarity_features.string_similarity import get_string_similarity
from src.utils import get_queries, get_targets, get_correct_targets, get_predicted_queries_and_targets_df, \
    load_pickled_object, decompress_file, pickle_object, compress_file, get_candidate_targets

sentence_embedding_models = ["distiluse-base-multilingual-cased-v1", "sentence-transformers/sentence-t5-base"]
referential_similarity_measures = ["synonym_similarity", "ne_similarity"]
lexical_similarity_measures = ["similar_words_ratio", "similar_words_ratio_length"]
string_similarity_measures = ["jaccard_similarity", "levenshtein", "sequence_matching"]
similarity_measure = "braycurtis"


def create_feature_target_correlation_file(data_pred, data_gold, target_file_path):
    pred_path = DATA_PATH + data_pred + "/pred_qrels.tsv"
    gold_path = DATA_PATH + data_gold + "/gold.tsv"
    query_path = DATA_PATH + data_gold +"/queries.tsv"
    corpus_path = target_file_path
    text_data_analysis_path = DATA_PATH + "evaluation/" + data_pred + "_text_data_analysis.tsv"

    queries = get_queries(query_path)
    targets = get_targets(corpus_path)
    original_target_ids = list(targets.keys())
    correct_targets = get_correct_targets(gold_path)
    pred_df = get_predicted_queries_and_targets_df(pred_path)

    pred_query_ids_not_unique = pred_df['query'].tolist()
    pred_query_ids = list(set(pred_df['query'].tolist()))
    pred_target_ids = pred_df['target'].tolist()

    candidate_targets = {k: targets[k] for k in pred_target_ids if k in targets}

    candidate_queries_and_targets = {}

    for query_id in pred_query_ids:
        candidate_queries_and_targets[query_id] = {}

    candidates = {}

    for index, row in pred_df.iterrows():
        query_id = row['query']
        candidates[query_id] = []

    for index, row in pred_df.iterrows():
        query_id = row['query']
        target_id = row['target']
        target_text = targets[target_id]
        candidate_queries_and_targets[query_id][target_id] = target_text
        candidates[query_id].append(target_id)

    candidate_targets = get_candidate_targets(candidates, targets)
    candidate_target_ids = list(candidate_targets.keys())

    all_sim_scores = {}
    all_features = []

    for query_id in pred_query_ids:
        all_sim_scores[query_id] = []

    caching_directory = DATA_PATH + "cache/" + data_pred

    for model in sentence_embedding_models:
        all_features.append(model)
        if "/" or ":" or "." in str(model):
            model_name = str(model).replace("/", "_").replace(":", "_").replace(".", "_")
        else:
            model_name = str(model)
        stored_embedded_queries = caching_directory + "/embedded_queries_" + model_name
        stored_embedded_targets = caching_directory + "/embedded_targets_" + model_name
        stored_sim_scores = caching_directory + "/sim_scores_" + model_name
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores + ".pickle" + ".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets, )
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
            if os.path.exists(stored_embedded_queries + ".pickle" + ".zip"):
                embedded_queries = load_pickled_object(decompress_file(stored_embedded_queries + ".pickle" + ".zip"))
            else:
                embedded_queries = encode_queries(queries, model)
                pickle_object(stored_embedded_queries, embedded_queries)
                compress_file(stored_embedded_queries + ".pickle")
                os.remove(stored_embedded_queries + ".pickle")
            if os.path.exists(stored_embedded_targets + ".pickle" + ".zip"):
                embedded_targets = load_pickled_object(decompress_file(stored_embedded_targets + ".pickle" + ".zip"))
                # keep only candidate targets
                embedded_candidate_targets = dict(
                    (k[embedded_targets]) for k in candidate_target_ids if k in embedded_targets)
            else:
                embedded_candidate_targets = encode_targets(candidate_targets, model)
            for query_id in list(queries.keys()):
                query_embedding = embedded_queries[query_id].reshape(1, -1)
                embedded_targets_array = np.array(list(embedded_candidate_targets.values()))
                sim_scores = (1 - cdist(query_embedding, embedded_targets_array,
                                        metric=similarity_measure)) * 100
                n_targets = sim_scores.shape[1]
                sim_scores = sim_scores.reshape(n_targets, )
                all_sim_scores[query_id].append(sim_scores)
    """
    2. For all referential similarity measures\
    2.1 get entities for all queries and cache or load from cache\
    2.2. get entities for all targets and cache or load from cache\
    2.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
    """
    for ref_feature in referential_similarity_measures:
        all_features.append(ref_feature)
        stored_entities_queries = caching_directory + "/queries_" + str(ref_feature)
        stored_entities_targets = caching_directory + "/targets_" + str(ref_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + ref_feature
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores + ".pickle" + ".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets, )
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
            if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
                entities_queries = load_pickled_object(decompress_file(stored_entities_queries + ".pickle" + ".zip"))
            else:
                entities_queries = get_sequence_entities(queries, ref_feature)
                print(entities_queries)
                pickle_object(stored_entities_queries, entities_queries)
                compress_file(stored_entities_queries + ".pickle")
                os.remove(stored_entities_queries + ".pickle")
            if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
                entities_targets = load_pickled_object(decompress_file(stored_entities_targets + ".pickle" + ".zip"))
                entities_candidate_targets = dict(
                    (k[entities_targets]) for k in candidate_target_ids if k in entities_targets)
            else:
                entities_candidate_targets = get_sequence_entities(candidate_targets, ref_feature)
            for query_id in list(queries.keys()):
                query_entities = set(entities_queries[query_id])
                sim_scores = np.zeros(len(list(candidate_targets.keys())))
                if query_entities:
                    len_query_entities = len(query_entities)
                    for idx, target_id in enumerate(list(candidate_targets.keys())):
                        target_entities = set(entities_candidate_targets[target_id])
                        len_target_entities = len(target_entities)
                        len_intersection = len(query_entities.intersection(target_entities))
                        ratio = (100 / (len_query_entities + len_target_entities)) * len_intersection
                        sim_scores[idx] = ratio
                all_sim_scores[query_id].append(sim_scores)
    """
    3. For all lexical similarity measures
    3.1 get entities for all queries and cache or load from cache\
    3.2. get entities for all targets and cache or load from cache\
    3.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
    """
    for lex_feature in lexical_similarity_measures:
        all_features.append(lex_feature)
        stored_entities_queries = caching_directory + "/queries_" + str(lex_feature)
        stored_entities_targets = caching_directory + "/targets_" + str(lex_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + lex_feature
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores + ".pickle" + ".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets, )
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
            if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
                entities_queries = load_pickled_object(decompress_file(stored_entities_queries + ".pickle" + ".zip"))
            else:
                entities_queries = get_lexical_entities(queries, lex_feature)
                pickle_object(stored_entities_queries, entities_queries)
                compress_file(stored_entities_queries + ".pickle")
                os.remove(stored_entities_queries + ".pickle")
            if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
                entities_targets = load_pickled_object(decompress_file(stored_entities_targets + ".pickle" + ".zip"))
                entities_candidate_targets = dict(
                    (k[entities_targets]) for k in candidate_target_ids if k in entities_targets)
            else:
                entities_candidate_targets = get_lexical_entities(candidate_targets, lex_feature)
            for query_id in list(queries.keys()):
                query_entities = set(entities_queries[query_id])
                sim_scores = np.zeros(len(list(candidate_targets.keys())))
                if query_entities:
                    len_query_entities = len(query_entities)
                    for idx, target_id in enumerate(list(candidate_targets.keys())):
                        target_entities = set(entities_candidate_targets[target_id])
                        len_target_entities = len(target_entities)
                        len_intersection = len(query_entities.intersection(target_entities))
                        len_union = len_query_entities + len_target_entities
                        if lex_feature == "similar_words_ratio":
                            ratio = (100 / (len_union / 2)) * len_intersection
                        elif lex_feature == "similar_words_ratio_length":
                            ratio = (100 / len_union) * len_intersection
                        sim_scores[idx] = ratio
                all_sim_scores[query_id].append(sim_scores)
    """
    4. For all string similarity measures
        4.1 Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
    """
    for string_feature in string_similarity_measures:
        all_features.append(string_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + string_feature
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores + ".pickle" + ".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets, )
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
            for query_id in list(queries.keys()):
                sim_scores = np.zeros(len(list(candidate_targets.keys())))
                for idx, target_id in enumerate(list(candidate_targets.keys())):
                    query = queries[query_id]
                    target = candidate_targets[target_id]
                    sim_scores[idx] = get_string_similarity(query, target, string_feature)
                all_sim_scores[query_id].append(sim_scores)



    columns = ['query_id', 'target_id', 'query', 'target', 'correct_pair']
    columns2 = ['correct_pair']

    for feature in all_features:
        columns.append(feature)
        columns2.append(feature)

    text_data_analysis_df = pd.DataFrame(columns=columns)

    corr_analysis = []
    correct_predicted = False

    target_idx = 0
    old_query_id = pred_query_ids_not_unique[0]

    for idx, query_id in enumerate(pred_query_ids_not_unique):
        if query_id != old_query_id:
            if not correct_predicted:
                if type(correct_target) != list:
                    correct_target = [correct_target]
                for c_target in correct_target:
                    c_target_text = targets[c_target]
                    query_text = queries[old_query_id]
                    this_row_scores = ['NOT PREDICTED_'+str(old_query_id), c_target, query_text, c_target_text, True]
                    for feature in all_features:
                        this_row_scores.append(get_sim_score(feature, query_text, c_target_text, similarity_measure))
                    this_row_df = pd.DataFrame([this_row_scores], columns=columns)
                    text_data_analysis_df = pd.concat([text_data_analysis_df, this_row_df])
            correct_predicted = False
            target_idx = 0
            old_query_id = query_id
            ending_line = ['_', '_', '_', '_', '_']
            for _ in all_features:
                ending_line.append('-')
            ending_line_df = pd.DataFrame([ending_line], columns=columns)
            text_data_analysis_df = pd.concat([text_data_analysis_df, ending_line_df])
        query = queries[query_id]
        target_id = pred_target_ids[idx]
        target = targets[target_id]
        sim_scores = []
        for feature_n in range(len(all_features)):
            sim_scores.append(all_sim_scores[query_id][feature_n][target_idx])
        correct_target = correct_targets[query_id]
        if type(correct_target) == list and target_id in correct_target:
            correct_pair = True
        else:
            if correct_target == target_id:
                correct_pair = True
                correct_predicted = True
            else:
                correct_pair = False
        this_row_correlation = [int(correct_pair)]
        this_row_scores = [query_id, target_id, query, target, correct_pair]
        for sim_score in sim_scores:
            this_row_correlation.append(sim_score)
            this_row_scores.append(round(sim_score, 3))
        corr_analysis.append(this_row_correlation)
        this_row_df = pd.DataFrame([this_row_scores], columns=columns)
        text_data_analysis_df = pd.concat([text_data_analysis_df, this_row_df])
        target_idx = target_idx + 1

    analyse_feature_correlation(columns2, np.array(corr_analysis), 'spearmanr', data_pred)

    text_data_analysis_df.to_csv(text_data_analysis_path, index=False, header=True, sep='\t')


path = DATA_PATH + 'sv_ident_val' + "/preprocessed/" + "study_title_variable_label_question_text_question_text_en_sub_question_item_categories_targets.tsv"
create_feature_target_correlation_file('sv_ident_val_fields_1', 'sv_ident_val', path)









