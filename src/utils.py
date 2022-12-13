import csv
import shutil
import gzip
import os
import json
import pickle
import pandas as pd
import numpy as np

from nltk import word_tokenize


def compress_file(filename_in):
    filename_out = filename_in+".zip"
    with open(filename_in, "rb") as fin, gzip.open(filename_out, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def decompress_file(comp_filename):
    with gzip.open(comp_filename, "rb") as fin:
        return fin.read()


def compress_directory(filename_in):
    shutil.make_archive(filename_in+".zip", 'zip', filename_in)


def pickle_object(filename, file):
    with open(filename+".pickle", 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickled_object(file_object):
    return pickle.loads(file_object)


def get_queries(query_path):
    df = pd.read_csv(query_path, sep = '\t', names=['id', 'query'], dtype = str)
    return df.set_index('id')['query'].to_dict()


def get_predicted_queries_and_targets_df(query_path):
    df = pd.read_csv(query_path, sep = '\t', names=['query', 'Q0', 'target', '1', 'score', 'tag'], dtype = str, index_col=False)
    return df


def get_correct_targets(query_path):
    df = pd.read_csv(query_path, sep = '\t', names=['query', '0', 'target', '1'], dtype = str)
    return df.set_index('query')['target'].to_dict()


def get_targets(corpus_of_targets_filename):
    if os.path.isdir(corpus_of_targets_filename):
        targets = {}
        for json_file in os.listdir(corpus_of_targets_filename):
            json_file_path = corpus_of_targets_filename + '/' + json_file
            with open(json_file_path, 'r', encoding='utf-8') as j:
                v_claim = json.loads(j.read())
            targets[v_claim['vclaim_id']] = v_claim['vclaim']
    else:
        df = pd.read_csv(corpus_of_targets_filename, sep='\t', dtype=str)
        column_length = len(df.columns)
        column_values = list(map(' '.join, df.iloc[:, 1:column_length].astype(str).values.tolist()))
        targets = dict(zip(df.iloc[:, 0], column_values))
    return targets


def get_certain_target_fields(corpus_of_targets_filename, fields):
    df = pd.read_csv(corpus_of_targets_filename, sep='\t', dtype=str)
    return df[fields]


def get_candidate_targets(candidates_dict, targets_dict):
    candidate_targets = {}
    for targets in candidates_dict.values():
        for target in targets:
            candidate_targets[target] = targets_dict[target]
    return candidate_targets



def make_top_k_dictionary(query_ids, target_ids, sim_scores_top_k_values, sim_scores_top_k_idx):
    top_k_targets_per_query = {}
    for query_itr in range(len(query_ids)):
        query_id = query_ids[query_itr]
        top_k_target_dict = {}
        for itr in range(len(sim_scores_top_k_idx[query_itr])):
            top_k_target_dict[target_ids[sim_scores_top_k_idx[query_itr][itr]]] = sim_scores_top_k_values[query_itr][itr]
        top_k_targets_per_query[query_id] = top_k_target_dict
    return top_k_targets_per_query


def candidates_dict_to_pred_qrels(candidates_dictionary, output_data_name):
    df = pd.DataFrame(columns=['qid', 'Q0', 'docno', 'rank', 'score', 'tag'])
    list_of_qids = []
    list_of_docno = []
    list_of_scores = []
    for qid, targets in candidates_dictionary.items():
        for target_id in targets:
            list_of_qids.append(qid)
            list_of_docno.append(target_id)
            list_of_scores.append(100)
    array_of_scores = np.array(list_of_scores)
    df = df.reset_index(drop=True)
    df['qid'] = pd.Series(list_of_qids)
    df['docno'] = pd.Series(list_of_docno)
    df['score'] = pd.Series(array_of_scores).astype(float)
    df['Q0'] = 'Q0'
    df['rank'] = '1'
    df['tag'] = 'SimBa'
    df = df.reset_index(drop=True)
    df.to_csv(output_data_name, index=False, header=False, sep='\t')


def output_dict_to_pred_qrels(output_dictionary, output_data_name):
    df = pd.DataFrame(columns=['qid', 'Q0', 'docno', 'rank', 'score', 'tag'])
    list_of_qids = []
    list_of_docno = []
    list_of_scores = []
    for qid, ranked_targets in output_dictionary.items():
        for target_id, sim_score in ranked_targets.items():
            list_of_qids.append(qid)
            list_of_docno.append(target_id)
            list_of_scores.append(sim_score)
    array_of_scores = np.array(list_of_scores)
    df = df.reset_index(drop=True)
    df['qid'] = pd.Series(list_of_qids)
    df['docno'] = pd.Series(list_of_docno)
    df['score'] = pd.Series(array_of_scores).astype(float)
    df['Q0'] = 'Q0'
    df['rank'] = '1'
    df['tag'] = 'SimBa'
    df = df.reset_index(drop=True)
    df.to_csv(output_data_name, index=False, header=False, sep='\t')


def delete_first_line_of_tsv(csv_file_name):
    df = pd.read_csv(csv_file_name, sep='\t', dtype=str)
    df.to_csv(csv_file_name, index=False, header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar=' ')


def prepare_corpus_tsv(corpus_path):
    df = pd.read_csv(corpus_path, sep='\t', dtype=str)
    df.rename(columns={df.columns[0]: "id"}, inplace=True)
    df.to_csv(corpus_path, index=False, header=True, sep='\t')


def append_qrels_documents(doc_1, doc_2, output_doc, header):
    if header:
        df1 = pd.read_csv(doc_1, sep='\t', header=True, dtype=str)
        df2 = pd.read_csv(doc_2, sep='\t', dtype=str)
    else:
        df1 = pd.read_csv(doc_1, sep='\t', dtype=str, names= ['qid', 'Q0', 'docno', 'rank', 'score', 'tag'])
        df2 = pd.read_csv(doc_2, sep='\t', dtype=str, names = ['qid', 'Q0', 'docno', 'rank', 'score', 'tag'])
    df3 = pd.concat([df2, df1], axis=0, keys = ['qid', 'Q0', 'docno', 'rank', 'score', 'tag'])
    if header:
        df3.to_csv(output_doc, index=False, header=True, sep='\t')
    else:
        df3.to_csv(output_doc, index=False, header=False, sep='\t')


def get_number_of_tokens(sequence):
    return len(word_tokenize(sequence))




















