import argparse
import os
import numpy as np
import torch
import pandas as pd

from src.create_similarity_features.lexical_similarity import get_lexical_entities
from src.create_similarity_features.referential_similarity import get_sequence_entities
from src.create_similarity_features.sentence_encoder import encode_queries, encode_targets
from src.candidate_retrieval import DATA_PATH
from src.create_similarity_features.string_similarity import get_string_similarity
from src.learning.create_feature_set import create_feature_set
from src.utils import load_pickled_object, decompress_file, get_queries, get_targets, pickle_object, compress_file, \
    get_number_of_tokens, make_top_k_dictionary, get_candidate_targets, output_dict_to_pred_qrels
from scipy.spatial.distance import cdist
from pathlib import Path




def run():
    """
    input:
    queries, targets, {query: list of top k targets (ordered if union is not chosen)}
    output:
    {query: list of top k targets (ordered if union is not chosen)}

    all_sim_scores: {query_id: list_of_sim_scores, list entries are arrays of shape (1, target_n) with target_n similarity scores between 0 and 100}
    """
    parser = argparse.ArgumentParser()
    #input
    parser.add_argument('queries', type=str, help='Input queries path as tsv file.')
    parser.add_argument('targets', type=str, help='Input targets path as tsv file.')
    # parameters
    parser.add_argument('data', type=str, help='Name under which the documents should be stored.')
    parser.add_argument('similarity_measure', type=str, default='braycurtis', help='Distance measure for sentence embeddings')
    parser.add_argument('k', type=int, default=100, help='How many targets per queries should be retrieved')
    parser.add_argument('--supervised', action="store_true",
                        help='If selected re-ranking is based on learning true pairs using the similarity features of the training data')
    parser.add_argument('-sentence_embedding_models', type=str, nargs='+',
                    default=[],
                    help='Pass a list of sentence embedding models hosted by Huggingface or Tensorflow or simply pass "infersent" to use the infersent encoder.')
    parser.add_argument('-referential_similarity_measures', type=str, nargs='+',
                        default=[])
    parser.add_argument('-lexical_similarity_measures', type=str, nargs='+', default=[],
                        help='Pass a list of lexical similarity measures to use')
    parser.add_argument('-string_similarity_measures', type=str, nargs='+', default=[])
    args = parser.parse_args()
    """
    Name datapaths and load queries and targets.
    queries: {query_id: text}
    targets: {target_id: text}
    candidates: {query_id: list of candidate target ids}
    candidate_targets: {candidate_target_id (taken from all possible canidates): target_text}
    ---
    possibly stored files:
    stored_sim_scores: {query_id: list of sim scores for all targets in order of original targets}
    """
    caching_directory = DATA_PATH + "cache/" + args.data
    Path(caching_directory).mkdir(parents=True, exist_ok=True)
    queries = get_queries(args.queries)
    targets = get_targets(args.targets)
    original_target_ids = list(targets.keys())
    candidates_path = DATA_PATH + args.data + "/candidates"
    candidates = load_pickled_object(decompress_file(candidates_path+".pickle"+".zip")) #
    candidate_targets = get_candidate_targets(candidates, targets)
    candidate_target_ids = list(candidate_targets.keys())
    all_features = []
    all_sim_scores = {}
    for query_id in list(queries.keys()):
        all_sim_scores[query_id] = []
    output_path = DATA_PATH + args.data + "/pred_qrels.tsv"
    """
    0. Learning
    """
    training_df = create_feature_set(args.data, targets, args.similarity_measure, args.sentence_embedding_models, args.referential_similarity_measures,
                       args.lexical_similarity_measures, args.string_similarity_measures)
    print(training_df)

    """
    1. For all sentence embedding models\
    1.1 Embed all queries and cache or load from cache\
    1.2. Embed all *relevant targets* or load from cache\
    1.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache   
    """
    for model in args.sentence_embedding_models:
        all_features.append(model)
        if "/" or ":" or "." in str(model):
            model_name = str(model).replace("/", "_").replace(":", "_").replace(".", "_")
        else:
            model_name = str(model)
        stored_embedded_queries = caching_directory + "/embedded_queries_" + model_name
        stored_embedded_targets = caching_directory + "/embedded_targets_" + model_name
        stored_sim_scores = caching_directory + "/sim_scores_" + model_name
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets,)
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
            if os.path.exists(stored_embedded_queries + ".pickle" + ".zip"):
                embedded_queries = load_pickled_object(decompress_file(stored_embedded_queries+".pickle"+".zip"))
            else:
                embedded_queries = encode_queries(queries, model)
                pickle_object(stored_embedded_queries, embedded_queries)
                compress_file(stored_embedded_queries + ".pickle")
                os.remove(stored_embedded_queries + ".pickle")
            if os.path.exists(stored_embedded_targets + ".pickle" + ".zip"):
                embedded_targets = load_pickled_object(decompress_file(stored_embedded_targets+".pickle"+".zip"))
                # keep only candidate targets
                embedded_candidate_targets = dict((k[embedded_targets]) for k in candidate_target_ids if k in embedded_targets)
            else:
                embedded_candidate_targets = encode_targets(candidate_targets, model)
            for query_id in list(queries.keys()):
                query_embedding = embedded_queries[query_id].reshape(1, -1)
                embedded_targets_array = np.array(list(embedded_candidate_targets.values()))
                sim_scores = (1 - cdist(query_embedding, embedded_targets_array,
                                        metric=args.similarity_measure)) * 100
                n_targets = sim_scores.shape[1]
                sim_scores = sim_scores.reshape(n_targets,)
                all_sim_scores[query_id].append(sim_scores)
    """
    2. For all referential similarity measures\
    2.1 get entities for all queries and cache or load from cache\
    2.2. get entities for all targets and cache or load from cache\
    2.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
    """
    for ref_feature in args.referential_similarity_measures:
        all_features.append(ref_feature)
        stored_entities_queries = caching_directory + "/queries_" + str(ref_feature)
        stored_entities_targets = caching_directory + "/targets_" + str(ref_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + ref_feature
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets,)
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
            if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
                entities_queries = load_pickled_object(decompress_file(stored_entities_queries+".pickle"+".zip"))
            else:
                entities_queries = get_sequence_entities(queries, ref_feature)
                print(entities_queries)
                pickle_object(stored_entities_queries, entities_queries)
                compress_file(stored_entities_queries + ".pickle")
                os.remove(stored_entities_queries + ".pickle")
            if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
                entities_targets = load_pickled_object(decompress_file(stored_entities_targets+".pickle"+".zip"))
                entities_candidate_targets = dict((k[entities_targets]) for k in candidate_target_ids if k in entities_targets)
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
                        ratio = (100/(len_query_entities+len_target_entities))*len_intersection
                        sim_scores[idx] = ratio
                all_sim_scores[query_id].append(sim_scores)
    """
    3. For all lexical similarity measures
    3.1 get entities for all queries and cache or load from cache\
    3.2. get entities for all targets and cache or load from cache\
    3.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
    """
    for lex_feature in args.lexical_similarity_measures:
        all_features.append(lex_feature)
        stored_entities_queries = caching_directory + "/queries_" + str(lex_feature)
        stored_entities_targets = caching_directory + "/targets_" + str(lex_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + lex_feature
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets,)
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
            if os.path.exists(stored_entities_queries + ".pickle" + ".zip"):
                entities_queries = load_pickled_object(decompress_file(stored_entities_queries+".pickle"+".zip"))
            else:
                entities_queries = get_lexical_entities(queries, lex_feature)
                pickle_object(stored_entities_queries, entities_queries)
                compress_file(stored_entities_queries + ".pickle")
                os.remove(stored_entities_queries + ".pickle")
            if os.path.exists(stored_entities_targets + ".pickle" + ".zip"):
                entities_targets = load_pickled_object(decompress_file(stored_entities_targets+".pickle"+".zip"))
                entities_candidate_targets = dict((k[entities_targets]) for k in candidate_target_ids if k in entities_targets)
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
                        len_union = len_query_entities+len_target_entities
                        if lex_feature == "similar_words_ratio":
                            ratio = (100/(len_union/2))*len_intersection
                        elif lex_feature == "similar_words_ratio_length":
                            ratio = (100/len_union)*len_intersection
                        sim_scores[idx] = ratio
                all_sim_scores[query_id].append(sim_scores)
    """
    4. For all string similarity measures
        4.1 Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
    """
    for string_feature in args.string_similarity_measures:
        all_features.append(string_feature)
        stored_sim_scores = caching_directory + "/sim_scores_" + string_feature
        if os.path.exists(stored_sim_scores + ".pickle" + ".zip"):
            sim_scores_to_store = load_pickled_object(decompress_file(stored_sim_scores+".pickle"+".zip"))
            for query_id in list(queries.keys()):
                current_candidate_ids = candidates[query_id]
                current_sim_scores = sim_scores_to_store[query_id]
                sim_scores_dict = dict(zip(original_target_ids, current_sim_scores))
                n_targets = len(current_candidate_ids)
                c_list = [sim_scores_dict[target_id] for target_id in current_candidate_ids]
                current_candidate_sim_scores = np.array(c_list).reshape(n_targets,)
                all_sim_scores[query_id].append(current_candidate_sim_scores)
        else:
            for query_id in list(queries.keys()):
                sim_scores = np.zeros(len(list(candidate_targets.keys())))
                for idx, target_id in enumerate(list(candidate_targets.keys())):
                    query = queries[query_id]
                    target = candidate_targets[target_id]
                    sim_scores[idx] = get_string_similarity(query, target, string_feature)
                all_sim_scores[query_id].append(sim_scores)
    """
    Evaluation step:
    Get mean and variance of all different similarity scores to better understand how to normalize them
    """
    all_sim_scores_df = pd.DataFrame.from_dict(all_sim_scores, orient='index', columns=all_features )
    for feature in all_features:
        print(feature)
        sim_scores = all_sim_scores_df[feature].to_numpy()
        sim_scores = np.hstack(sim_scores)
        sim_scores_mean = np.mean(sim_scores, axis=0)
        print(sim_scores_mean)
    """
    5. get top k targets per query:
    5.1. either supervised using the model trained above
    5.2. or compute mean of features and compute top k  
    """
    output = {}
    if args.supervised:
        pass
    else:
        for query_id, query_sim_scores in list(all_sim_scores.items()):
            mean_sim_scores = np.mean(query_sim_scores, axis=0)
            targets_and_sim_scores = dict(zip(candidates[query_id], mean_sim_scores))
            targets_and_sim_scores = dict(
                sorted(targets_and_sim_scores.items(), key=lambda item: item[1], reverse=True))
            targets_and_sim_scores = {x: targets_and_sim_scores[x] for x in list(targets_and_sim_scores)[:args.k]}
            output[query_id] = targets_and_sim_scores

    print(output)

    output_dict_to_pred_qrels(output, output_path)


if __name__ == "__main__":
    run()