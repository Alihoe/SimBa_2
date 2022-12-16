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
    #                  "sv_ident_val",
    #                 '-fields', 'question_text'])


    ## Val 1 Certain Fields


    subprocess.call(["python",
                     "../../src/candidate_retrieval/retrieval.py",
                     "../../data/sv_ident_val/queries.tsv",
                     "../../data/sv_ident_val/preprocessed/study_title_variable_label_question_text_question_text_en_sub_question_item_categories_targets.tsv",
                     "sv_ident_val_fields_1",
                     "braycurtis",
                     "5000",
                     "--union_of_top_k_per_feature",
                     '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1",
                     "princeton-nlp/sup-simcse-roberta-large", "infersent", "all-mpnet-base-v2"
                     "https://tfhub.dev/google/universal-sentence-encoder/4"
                     '-referential_similarity_measures', "synonym_similarity", "ne_similarity",
                     '-lexical_similarity_measures', "similar_words_ratio", "similar_words_ratio_length",
                     '-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])

    subprocess.call(["python", "../../evaluation/scorer/recall_evaluator.py",
                     "sv_ident_val_fields_1",
                     "../../data/sv_ident_val/gold.tsv"])

    subprocess.call(["python",
                     "../../src/re_ranking/re_ranking.py",
                     "../../data/sv_ident_val/queries.tsv",
                     "../../data/sv_ident_val/preprocessed/study_title_variable_label_question_text_question_text_en_sub_question_item_categories_targets.tsv",
                     "sv_ident_val_fields_1",
                     "braycurtis",
                     "10",
                     '-sentence_embedding_models', "sentence-transformers/sentence-t5-base", "distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2",
                     "princeton-nlp/sup-simcse-roberta-large", "infersent", "https://tfhub.dev/google/universal-sentence-encoder/4"])
                     #'-referential_similarity_measures', "synonym_similarity", "ne_similarity",
                     #'-lexical_similarity_measures', "similar_words_ratio_", "similar_words_ratio_legth",
                     #'-string_similarity_measures', "jaccard_similarity", "levenshtein", "sequence_matching"])

    subprocess.call(["python", "../../evaluation/scorer/main.py",
                     "sv_ident_val_fields_1",
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