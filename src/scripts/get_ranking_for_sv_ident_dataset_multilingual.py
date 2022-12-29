import subprocess


def run():

    ## SV Ident Val



    # subprocess.call(["python",
    #                  "../../src/candidate_retrieval/retrieval.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val/corpus",
    #                  "sv_ident_val",
    #                  "braycurtis",
    #                  "5000",
    #                  "--union_of_top_k_per_feature",
    #                  #'-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1",
    #                  # '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  # '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  '-string_similarity_measures', "sequence_matching"])#, "levenshtein", "sequence_matching"])
    #
    # subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
    #                  "sv_ident_val",
    #                  "../../data/sv_ident_val/gold.tsv"])

    # subprocess.call(["python",
    #                  "../../src/re_ranking/re_ranking.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val/corpus",
    #                  "sv_ident_val",
    #                  "braycurtis",
    #                  "10",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base"])#,"distiluse-base-multilingual-cased-v1",
    #                  # '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  # '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  # '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    #
    # subprocess.call(["python", "../../evaluation/scorer/main.py",
    #                  "sv_ident_val",
    #                  "../../data/sv_ident_val/gold.tsv"])

    ## Example Pre-processing

    # subprocess.call(["python",
    #                  "../../src/pre_processing/pre_processing_targets.py",
    #                  "../../data/sv_ident_val/corpus",
    #                  "sv_ident_val_variable_label",
    #                 '-fields', 'variable_label'])

    # data_name = "sv_ident_val_variable_label"
    #
    # subprocess.call(["python",
    #                  "../../src/candidate_retrieval/retrieval.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_variable_label/variable_label_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "5000",
    #                  "--union_of_top_k_per_feature",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1",
    #                  "princeton-nlp/sup-simcse-roberta-large", "all-mpnet-base-v2", "infersent",
    #                  "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
    #                  '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    #
    # subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    #
    # subprocess.call(["python",
    #                  "../../src/re_ranking/re_ranking.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_variable_label/variable_label_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "10",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2",
    #                  "princeton-nlp/sup-simcse-roberta-large", "infersent"])#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
    #                  #'-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  #'-lexical_similarity_measures', "similar_words_ratio_", "similar_words_ratio_legth",
    #                  #'-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    #
    # subprocess.call(["python", "../../evaluation/scorer/main.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])

    # Only question text

    # data_name = "sv_ident_val_question_text"
    #
    # subprocess.call(["python",
    #                  "../../src/candidate_retrieval/retrieval.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_question_text/question_text_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "5000",
    #                  "--union_of_top_k_per_feature",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1",
    #                  "princeton-nlp/sup-simcse-roberta-large", "all-mpnet-base-v2", "infersent",
    #                  #"https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
    #                  '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    #
    # subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    #
    # subprocess.call(["python",
    #                  "../../src/re_ranking/re_ranking.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_question_text/question_text_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "10",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2",
    #                  "princeton-nlp/sup-simcse-roberta-large", "infersent"])#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
    #                  #'-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  #'-lexical_similarity_measures', "similar_words_ratio_", "similar_words_ratio_legth",
    #                  #'-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    #
    # subprocess.call(["python", "../../evaluation/scorer/main.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    
    # data_name = "sv_ident_val_study_title"
    # 
    # subprocess.call(["python",
    #                  "../../src/pre_processing/pre_processing_targets.py",
    #                  "../../data/sv_ident_val/corpus",
    #                  "sv_ident_val_study_title",
    #                 '-fields', 'study_title'])
    # 
    # subprocess.call(["python",
    #                  "../../src/candidate_retrieval/retrieval.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_study_title/study_title_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "5000",
    #                  "--union_of_top_k_per_feature",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1",
    #                  "princeton-nlp/sup-simcse-roberta-large", "all-mpnet-base-v2", "infersent",
    #                  #"https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
    #                  '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    # 
    # subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    # 
    # subprocess.call(["python",
    #                  "../../src/re_ranking/re_ranking.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_study_title/study_title_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "10",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2",
    #                  "princeton-nlp/sup-simcse-roberta-large", "infersent"])#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
    #                  #'-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  #'-lexical_similarity_measures', "similar_words_ratio_", "similar_words_ratio_legth",
    #                  #'-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    # 
    # subprocess.call(["python", "../../evaluation/scorer/main.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    
    # data_name = "sv_ident_val_question_text_en"
    # 
    # subprocess.call(["python",
    #                  "../../src/pre_processing/pre_processing_targets.py",
    #                  "../../data/sv_ident_val/corpus",
    #                  "sv_ident_val_question_text_en",
    #                 '-fields', 'question_text_en'])
    # 
    # subprocess.call(["python",
    #                  "../../src/candidate_retrieval/retrieval.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_question_text_en/question_text_en_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "5000",
    #                  "--union_of_top_k_per_feature",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1",
    #                  "princeton-nlp/sup-simcse-roberta-large", "all-mpnet-base-v2", "infersent",
    #                  #"https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
    #                  '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    # 
    # subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    # 
    # subprocess.call(["python",
    #                  "../../src/re_ranking/re_ranking.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_question_text_en/question_text_en_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "10",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2",
    #                  "princeton-nlp/sup-simcse-roberta-large", "infersent"])#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
    #                  #'-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  #'-lexical_similarity_measures', "similar_words_ratio_", "similar_words_ratio_legth",
    #                  #'-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    # 
    # subprocess.call(["python", "../../evaluation/scorer/main.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    
    # data_name = "sv_ident_val_sub_question"
    # 
    # subprocess.call(["python",
    #                  "../../src/pre_processing/pre_processing_targets.py",
    #                  "../../data/sv_ident_val/corpus",
    #                  "sv_ident_val_sub_question",
    #                 '-fields', 'sub_question'])
    # 
    # subprocess.call(["python",
    #                  "../../src/candidate_retrieval/retrieval.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_sub_question/sub_question_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "5000",
    #                  "--union_of_top_k_per_feature",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1",
    #                  "princeton-nlp/sup-simcse-roberta-large", "all-mpnet-base-v2", "infersent",
    #                  #"https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
    #                  '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    # 
    # subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    # 
    # subprocess.call(["python",
    #                  "../../src/re_ranking/re_ranking.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_sub_question/sub_question_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "10",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2",
    #                  "princeton-nlp/sup-simcse-roberta-large", "infersent"])#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
    #                  #'-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  #'-lexical_similarity_measures', "similar_words_ratio_", "similar_words_ratio_legth",
    #                  #'-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    # 
    # subprocess.call(["python", "../../evaluation/scorer/main.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    
    # data_name = "sv_ident_val_item_categories"
    # 
    # subprocess.call(["python",
    #                  "../../src/pre_processing/pre_processing_targets.py",
    #                  "../../data/sv_ident_val/corpus",
    #                  "sv_ident_val_item_categories",
    #                 '-fields', 'item_categories'])
    # 
    # subprocess.call(["python",
    #                  "../../src/candidate_retrieval/retrieval.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_item_categories/item_categories_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "5000",
    #                  "--union_of_top_k_per_feature",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1",
    #                  "princeton-nlp/sup-simcse-roberta-large", "all-mpnet-base-v2", "infersent",
    #                  #"https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
    #                  '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    # 
    # subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    # 
    # subprocess.call(["python",
    #                  "../../src/re_ranking/re_ranking.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_item_categories/item_categories_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "10",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2",
    #                  "princeton-nlp/sup-simcse-roberta-large", "infersent"])#, "https://tfhub.dev/google/universal-sentence-encoder/4"])
    #                  #'-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  #'-lexical_similarity_measures', "similar_words_ratio_", "similar_words_ratio_legth",
    #                  #'-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    # 
    # subprocess.call(["python", "../../evaluation/scorer/main.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])

    # data_name = "sv_ident_val_question_text_question_text_en"
    #
    # subprocess.call(["python",
    #                  "../../src/pre_processing/pre_processing_targets.py",
    #                  "../../data/sv_ident_val/corpus",
    #                  "sv_ident_val_question_text_question_text_en",
    #                  '-fields', 'question_text', 'question_text_en'])
    #
    # subprocess.call(["python",
    #                  "../../src/candidate_retrieval/retrieval.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_question_text_question_text_en/question_text_question_text_en_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "5000",
    #                  "--union_of_top_k_per_feature",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base",
    #                  "distiluse-base-multilingual-cased-v1",
    #                  "princeton-nlp/sup-simcse-roberta-large", "all-mpnet-base-v2", "infersent",
    #                  # "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
    #                  '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    #
    # subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    #
    # subprocess.call(["python",
    #                  "../../src/re_ranking/re_ranking.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_question_text_question_text_en/question_text_question_text_en_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "10",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base",
    #                  "distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2",
    #                  "princeton-nlp/sup-simcse-roberta-large",
    #                  "infersent"])  # , "https://tfhub.dev/google/universal-sentence-encoder/4"])
    # # '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    # # '-lexical_similarity_measures', "similar_words_ratio_", "similar_words_ratio_legth",
    # # '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    #
    # subprocess.call(["python", "../../evaluation/scorer/main.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])


    ## Val 1 Certain Fields

    data_name = "sv_ident_val_fields_1"
    # 
    # subprocess.call(["python",
    #                  "../../src/candidate_retrieval/retrieval.py",
    #                  "../../data/sv_ident_val/queries.tsv",
    #                  "../../data/sv_ident_val_fields_1/study_title_variable_label_question_text_question_text_en_sub_question_item_categories_pp_targets.tsv",
    #                  data_name,
    #                  "braycurtis",
    #                  "5000",
    #                  "--union_of_top_k_per_feature",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1",
    #                  "princeton-nlp/sup-simcse-roberta-large", "all-mpnet-base-v2", "infersent",
    #                  "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
    #                  '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    # 
    # subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
    #                  data_name,
    #                  "../../data/sv_ident_val/gold.tsv"])
    # 
    subprocess.call(["python",
                     "../../src/re_ranking/re_ranking.py",
                     "../../data/sv_ident_val/queries.tsv",
                     "../../data/sv_ident_val_fields_1/study_title_variable_label_question_text_question_text_en_sub_question_item_categories_pp_targets.tsv",
                     data_name,
                     "braycurtis",
                     "10",
                     'supervised',
                     '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2",
                     "princeton-nlp/sup-simcse-roberta-large", "infersent", "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
                     '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
                     '-lexical_similarity_measures', "similar_words_ratio_", "similar_words_ratio_legth",
                     '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])

    subprocess.call(["python", "../../evaluation/scorer/main.py",
                     data_name,
                     "../../data/sv_ident_val/gold.tsv"])


    ## Trial Data


    # subprocess.call(["python",
    #                  "../../src/candidate_retrieval/retrieval.py",
    #                  "../../data/sv_ident_trial_en/queries.tsv",
    #                  "../../data/sv_ident_trial_en/corpus",
    #                  "sv_ident_trial_en",
    #                  "braycurtis",
    #                  "20",
    #                  "--union_of_top_k_per_feature",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1",
    #                  '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    #
    # subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
    #                  "sv_ident_trial_en",
    #                  "../../data/sv_ident_trial_en/gold.tsv"])
    #
    # subprocess.call(["python",
    #                  "../../src/re_ranking/re_ranking.py",
    #                  "../../data/sv_ident_trial_en/queries.tsv",
    #                  "../../data/sv_ident_trial_en/corpus",
    #                  "sv_ident_trial_en",
    #                  "braycurtis",
    #                  "10",
    #                  '-sentence_embedding_models', "sentence-transformers/sentence-t5-base"])#, "distiluse-base-multilingual-cased-v1",
    #                  # '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
    #                  # '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
    #                  # '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])
    #
    # subprocess.call(["python", "../../evaluation/scorer/main.py",
    #                  "sv_ident_trial_en",
    #                  "../../data/sv_ident_trial_en/gold.tsv"])



if __name__ == "__main__":
    run()