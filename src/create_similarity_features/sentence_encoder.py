import os

from sentence_transformers import SentenceTransformer
import logging
import nltk
import torch
import tensorflow_hub as hub

from src.create_similarity_features.infersent_encoder.infersent.models import InferSent

logger = logging.getLogger(__name__)


def encode_queries(query_dictionary, model_name):
    batch_size = 128
    queries = list(query_dictionary.values())
    ids = list(query_dictionary.keys())
    if model_name == "infersent":
        word_embedding_type = "fast_text"
        nltk.download('punkt')
        infersent_model_path = os.getcwd()+"/src/infersent_encoder/"
        if word_embedding_type == "glove":
            params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                            'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
            model = InferSent(params_model)
            model.load_state_dict(torch.load(infersent_model_path + "infersent/infersent1.pkl"))
            model.set_w2v_path(infersent_model_path + "glove.840B.300d.txt")
        elif word_embedding_type == "fast_text":
            params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                            'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
            model = InferSent(params_model)
            model.load_state_dict(torch.load(infersent_model_path + "infersent/infersent2.pkl"))
            model.set_w2v_path(infersent_model_path + "crawl-300d-2M.vec")
        model.build_vocab(queries, tokenize=True)
        encoded_queries = model.encode(queries, tokenize=True)
    elif model_name.startswith("https://tfhub.dev"):
        model = hub.load(model_name)
        encoded_queries = model(queries)
    else:
        encoded_queries = SentenceTransformer(model_name).encode(queries, batch_size)
    return dict(zip(ids, encoded_queries))


def encode_targets(target_dictionary, model_name):
    batch_size = 128
    chunk_size = 50000
    targets = list(target_dictionary.values())
    target_ids = list(target_dictionary.keys())
    itr = range(0, len(targets), chunk_size)
    encoded_targets = {}
    if model_name == "infersent":
        if model_name == "infersent":
            word_embedding_type = "fast_text"
            nltk.download('punkt')
            infersent_model_path = "../infersent_encoder/"
            if word_embedding_type == "glove":
                params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
                model = InferSent(params_model)
                model.load_state_dict(torch.load(infersent_model_path + "infersent/infersent1.pkl"))
                model.set_w2v_path(infersent_model_path + "glove.840B.300d.txt")
            elif word_embedding_type == "fast_text":
                params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                                'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
                model = InferSent(params_model)
                model.load_state_dict(torch.load(infersent_model_path + "infersent/infersent2.pkl"))
                model.set_w2v_path(infersent_model_path + "crawl-300d-2M.vec")
            model.build_vocab(targets, tokenize=True)
            for batch_num, corpus_start_idx in enumerate(itr):
                logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
                corpus_end_idx = min(corpus_start_idx + chunk_size, len(targets))
                sub_corpus_embeddings = model.encode(targets[corpus_start_idx:corpus_end_idx], batch_size)
                encoded_targets = encoded_targets | dict(zip(target_ids[corpus_start_idx:corpus_end_idx], sub_corpus_embeddings))
    elif model_name.startswith("https://tfhub.dev"):
        model = hub.load(model_name)
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + chunk_size, len(targets))
            sub_corpus_embeddings = model(targets[corpus_start_idx:corpus_end_idx])
            encoded_targets = encoded_targets | dict(
                zip(target_ids[corpus_start_idx:corpus_end_idx], sub_corpus_embeddings))
    else:
        model = SentenceTransformer(model_name)
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + chunk_size, len(targets))
            sub_corpus_embeddings = model.encode(targets[corpus_start_idx:corpus_end_idx], batch_size)
            encoded_targets = encoded_targets | dict(zip(target_ids[corpus_start_idx:corpus_end_idx], sub_corpus_embeddings))
    return encoded_targets




