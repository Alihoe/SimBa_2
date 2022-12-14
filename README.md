
# SimBa

## Arcitecture

1. Retrieval
2. Re-Ranking
3. Learning

### Retrieval

**input**:
queries, targets

**output**:
{query: list of top k targets (ordered if union is not chosen)}

**parameters**:
- data name for storage
- similarity measure for embeddings
- k
- *features*: sentence embeddings models, lexical similarity measures, referential similarity measures, string similarity measures, referntial similarity measures
- How to combine the features: either take mean of different features or union of top k per feature

**possibly cached**:
- sentence embeddings queries and targets (df)
- entities queries and targets (referential similarity) (df)
- similarity scores for all used features (dictionary)

#### Retrieval Architecture

0. Get number of tokens of\
    0.1 queries
    0.2 targets
    0.3 pairs

1. For all sentence embedding models\
   1.1 Embed all queries and cache\
   1.2. Embed all targets and cache\
   1.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
   
2. For all referential similarity measures\
   2.1 get entities for all queries and cache\
   2.2. get entities for all targets and cache\
   2.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
    
3. For all lexical similarity measures\
    3.1 get entities for all queries and cache or load from cache\
    3.2. get entities for all targets and cache or load from cache\
    3.3. Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
   
4. For all string similarity measures\
    4.1 Calculate all similarity scores for all combinations -> value between 0 and 100 and cache
   
5. get top k targets per query:\
    5.1. create union of features and compute top k\
    5.2. compute mean of features and compute top k   
   
### Re-Ranking

**input**:
{query: list of top k targets (ordered if union is not chosen)}, queries, targets

**output**:
qrels file with top k targets per query including similarity score

**parameters**:
- data name for storage
- similarity measure for embeddings
- k
- *features*: sentence embeddings models, lexical similarity measures, referential similarity measures, string similarity measures, referntial similarity measures
- How to combine the features:
    - either take mean of different features or
    - learn the similarity using the features
    
#### Re-Ranking Architecture

0. If similarity should be learned:
    Do Learning, then all target embeddings should be cached

1. For all sentence embedding models\
   1.1 Embed all queries and cache or load from cache\
   1.2. Embed all *relevant targets* or load from cache\
   1.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
   
2. For all referential similarity measures\
   2.1 get entities for all queries and cache or load from cache\
   2.2. get entities for all targets and cache or load from cache\
   2.3. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
   
3. For all lexical similarity measures\
    3.1. Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
   
4. For all string similarity measures\
    4.1 Calculate all similarity scores for one query and its *candidate targets* or load from cache -> value between 0 and 100 and cache
   
5. get top k targets per query:\
    5.1. compute mean of features and compute top k  
   5.2. predict relevance using trained model

### Learning

**problem**
learn to rank using binary training data
One approach: binary classification using classifiers certainty as similarity score

**input**
training queries, targets

**output**
model trained on input features

**parameters**
- data name for storage
- similarity measure for embeddings
- k for retrieval step 
- *features*: sentence embeddings models, lexical similarity measures, referential similarity measures, string similarity measures, referntial similarity measures

#### Learning Architecture

0. Do Retrieval step for training queries with same features as Re-Ranking
1. Create feature set with true pairs as targets, include query-target pairs that were not retrieved
2. Train a binary classifier

### Pre-Processing
#### Queries
**input**
queries
**output**
pre-processed queries
#### Targets
**input**
targets
**output**
pre-processed targets

##### Using particular fields

## Questions regarding architecture?

- Which features are the best for each step?
- Supervised or Unsupervised Re-Ranking?
- Use union of mean in retrieval?
- Are the features properly normalized?

## Analysis
- Which fields are the most useful?
- Multilinguality

# Evaluation

## - Which features are best for each step?

### Correlation Analysis

- Trial Data English

The correlation for feature correct_pair

with feature sentence-transformers/sentence-t5-base is 0.29

with feature synonym_similarity is 0.218

with feature distiluse-base-multilingual-cased-v1 is 0.178

with feature similar_words_ratio is 0.133

with feature similar_words_ratio_length is 0.133

with feature levenshtein is 0.103

with feature sequence_matching is 0.054

with feature jaccard_similarity is 0.052

with feature ne_similarity is 0.045

Which features to test: 

- Best results using sentence transformers and synonym similarity
- also try similar words ratio and levenshtein

### How to get a high recall after retrieval

Recall after Retrieval 

- Trial Data English

|Features|k=50|k=100|k=182|
|--------|----|-----|-----|
|distiluse-base-multilingual-cased-v1|0.9583|0.9792|1|
|sentence-transformers/sentence-t5-base|0.9792|0.9792|1|
|synonym_similarity|0.7708|0.9167|1|    
|ne_similarity|0.6042|0.8542|1|
|similar_words_ratio|0.7708|0.8750|1|
|similar_words_ratio_length|0.7708|0.8750|1|
|jaccard_similarity|0.5833|0.8542|1|
|levenshtein|0.5625|0.7917|1|
|sequence_matching|0.2917|0.5208|1|

Using all features as union

|k     |      |
|------|------|
|k = 10|0.9792|
|k= 20 |     1|

--> k = 20 is already sufficient

- Val Data using all fields

|Features|k=50|k=100|k=200|k=500|k=1000|k=5000|k=10,000|k=20,000|k=21328|
|--------|----|-----|-----|-----|------|------|--------|--------|-------|
|distiluse-base-multilingual-cased-v1|0.1128|0.1553|0.2089|0.3142|0.4214|0.7505|0.9353|0.9963|1|
|sentence-transformers/sentence-t5-base|0.1128|0.1497|0.2015|0.2773|0.3475|0.6506|0.7985|0.9908|1|
|synonym_similarity|0.0222|0.0314|0.0407|0.0665|0.1368|0.3346|0.5564|0.9556|1|
|ne_similarity|0.0425|0.0462|0.0647|0.0906|0.1516|0.5804|0.7172|0.9908|1|
|similar_words_ratio|0.0536|0.0776|0.1035|0.1479|0.1885|0.5213|0.6987|0.9445|1|
|similar_words_ratio_length|0.0536|0.0776|0.1035|0.1479|0.1885|0.5213|0.6987|0.9445|1|
|jaccard_similarity|0.0351|0.0481|0.0628|0.1109|0.1368|0.3549|0.4972|0.8983|1|
|levenshtein|0.0092|0.0129|0.0222|0.0499|0.0628|0.1996|0.4954|0.9261|1|
|sequence_matching|0.0166|0.0203|0.0388|0.0795|0.1072|0.3309|0.5323|0.9224|1|

Using all features as union

|k     |Recall|
|------|------|
|k = 50|0.2126|
|k = 100|0.2699|
|k = 200|0.3715|
|k = 500|0.4972|
|k = 1000|0.6488|
|k = 5000|**0.9353**|
|k = 10,000|**1**|
|k = 20,000|1|

--> 5000 is sufficient

- Val Data using only 'study_title', 'variable_label', 'question_text', 'question_text_en', 'sub_question', 'item_categories'

|Features|k=50|k=100|k=200|k=500|k=1000|k=5000|k=10,000|k=20,000|k=21328|
|--------|----|-----|-----|-----|------|------|--------|--------|-------|
|distiluse-base-multilingual-cased-v1|0.1386|0.1793|0.2218|0.3105|0.3826|0.6969|0.8281|0.9982|1|
|sentence-transformers/sentence-t5-base|0.1183|0.1497|0.2033|0.2921|0.3734|0.5693|0.7856|0.9945|1|
|synonym_similarity|0.0222|0.0314|0.0481|0.0776|0.1312|0.4067|0.5878|0.9649|1|
|ne_similarity|0.0296|0.0388|0.0518|0.0795|0.1349|0.6026|0.7320|0.9926|1|
|similar_words_ratio|0.0518|0.0665|0.0943|0.1516|0.2163|0.5749|0.7190|0.9482|1|
|similar_words_ratio_length|0.0518|0.0665|0.0943|0.1516|0.2163|0.5749|0.7190|0.9482|1|
|jaccard_similarity|0.0351|0.0370|0.0444|0.0943|0.1294|0.3475|0.5675|0.9538|1|
|levenshtein|0.0074|0.0092|0.0148|0.0481|0.0924|0.3327|0.6081|0.9797|1|
|sequence_matching|0.0111|0.0129|0.0148|0.0240|0.0481|0.2643|0.5730|0.9279|1|

Using all features as union

|k     |Recall|
|------|------|
|k = 50|0.2144|
|k = 100|0.2921|
|k = 200|0.3697|
|k = 500|0.4824|
|k = 1000|0.6044|
|k = 5000|**0.9224**|
|k = 10,000|0.9667|
|k = 20,000|**1**|

--> 5000 is sufficient

### How to get a high MAP after re-ranking?

Unsupervised arithmetic averaging of all features

- Trial Data English

|Features|MAP@10|
|--------|------|
|all     |0.5634|
|sentence-transformers/sentence-t5-base|**0.8076**|
|distiluse-base-multilingual-cased-v1|0.5959|
|synonym_similarity|0.3764|
|ne_similarity|0.1173|
|similar_words_ratio|0.4193|
|similar_words_ratio_length|0.4193|
|jaccard_similarity|0.2287|
|levenshtein|0.2195|
|sequence_matching|0.1062|
|sentence-transformers/sentence-t5-base + synonym_similarity|0.6565|
|sentence-transformers/sentence-t5-base + similar_words_ratio|0.7183|
|sentence-transformers/sentence-t5-base + similar_words_ratio_length|**0.7891**|
|sentence-transformers/sentence-t5-base + levenshtein|0.5250|
|sentence-transformers/sentence-t5-base + distiluse-base-multilingual-cased-v1|0.7235|

--> use strongest sentence transformer for re-ranking

- Val Data using all fields

|Features|MAP@10|
|--------|------|
|all     |0.0133|
|sentence-transformers/sentence-t5-base|**0.0307**|
|distiluse-base-multilingual-cased-v1|0.0199|
|synonym_similarity|0.0037|
|ne_similarity|0.0038|
|similar_words_ratio|0.0206|
|similar_words_ratio_length|0.0206|
|jaccard_similarity|0.0152|
|levenshtein|0.0014|
|sequence_matching| 0.0021|
|sentence-transformers/sentence-t5-base + synonym_similarity|0.0106|
|sentence-transformers/sentence-t5-base + similar_words_ratio|0.0181|
|sentence-transformers/sentence-t5-base + similar_words_ratio_length|0.0216|
|sentence-transformers/sentence-t5-base + levenshtein|0.0145|
|sentence-transformers/sentence-t5-base + distiluse-base-multilingual-cased-v1|0.0242|
|sentence-transformers/sentence-t5-base + distiluse-base-multilingual-cased-v1 + similar_words_ratio_length|0.0232|

--> use strongest sentence transformer for re-ranking

- Val Data using only 'study_title', 'variable_label', 'question_text', 'question_text_en', 'sub_question', 'item_categories'


|Features|MAP@10|
|--------|------|
|all     |0.0287|
|sentence-transformers/sentence-t5-base|0.0363|
|distiluse-base-multilingual-cased-v1|0.0358|
|synonym_similarity|0.0066|
|ne_similarity|0.0049|
|similar_words_ratio|0.0249|
|similar_words_ratio_length|0.0249
|jaccard_similarity|0.0123|
|levenshtein|0.0036|
|sequence_matching|0.0040|
|sentence-transformers/sentence-t5-base + synonym_similarity|0.0073|
|sentence-transformers/sentence-t5-base + similar_words_ratio|0.0342|
|sentence-transformers/sentence-t5-base + similar_words_ratio_length|**0.0398**|
|sentence-transformers/sentence-t5-base + levenshtein|0.0190|
|sentence-transformers/sentence-t5-base + distiluse-base-multilingual-cased-v1|**0.0449**|
|sentence-transformers/sentence-t5-base + distiluse-base-multilingual-cased-v1 + similar_words_ratio_length|**0.0447**|

--> combine sentence transformers for re-ranking








