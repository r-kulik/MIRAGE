# Параметризуемые сущности и где они обитают

Обновлено 29 мар. 2025 г., 19:39:06  
Информация  

## RawStorage  
Хранилище исходных документов  

### ChunkingAlgorithm  
Разделение больших документов на части (чанки) маленького текста  

#### WordCountingChunkingAlgorithm  
Алгоритм, основанный на подсчёте слов в чанке  

**Параметры:**  
| Parameter     | Type | Domain   |  
|--------------|------|----------|  
| words_amount | int  | 10 - 100 |  

#### SentenceChunkingAlgorithm  
Алгоритм, группирующий текст по предложениям  

**Параметры:**  
| Parameter          | Type | Domain |  
|-------------------|------|--------|  
| sentences_in_chunk | int  | 1-20   |  

#### SemanticChunkingAlgorithm  
Алгоритм, группирующий чанки по семантической близости  

**Параметры:**  
| Parameter        | Type  | Domain    |  
|-----------------|-------|-----------|  
| max_chunk_size  | int   | 1-100     |  
| threshold       | float | 0-1       |  

## ChunkStorage  
Хранение для полнотекстового поиска по чанкам  

### WhooshChunkStorage  
Хранение чанков на движке Whoosh  

**Параметры:**  
| Parameter           | Type      | Domain       |  
|--------------------|-----------|--------------|  
| scoring_function   | Choice    | tfidf, BM25F |  
| normalizer         | bool      | 1, 0         |  
| K1                 | float     | 0-3          |  
| B                  | float     | 0-1          |  

## FAISS Vector  
Хранение для поиска по векторным представлениям чанков  

### FaissIndexFlatL2  
Простой flat-индекс, полный перебор по норме L2 (евклидово расстояние)  

### FaissIndexFlatIP  
Простой flat-индекс, полный перебор по InnerProduct  

### FaissIndexHNSWFlat  
Индекс, основанный на Hierarchical Navigable Small World  

**Параметры:**  
| Parameter        | Type | Domain |  
|-----------------|------|--------|  
| M               | int  | 2-22?  |  
| efConstruction  | int  | 3-22?  |  

### FaissIndexIVFFlat  
Индекс, основанный на ячейках Вороного  

**Параметры:**  
| Parameter | Type        | Domain   |  
|----------|-------------|----------|  
| nlist    | int         | 127?     |  
| metric   | Categorical | L2, IP   |  

### FaissIndexLSH  
Local Sensitive Hashing  

**Параметры:**  
| Parameter | Type | Domain     |  
|----------|------|------------|  
| nbits    | int  | 2, 4, 8... |  

### FaissIndexScalarQuantizer  
Скалярная квантизация индексов  

**Параметры:**  
| Parameter | Type        | Domain          |  
|----------|-------------|-----------------|  
| quantizer | Quantizer   | 8, 6, 4_bit, ...|  
| metric    | Categorical | L2, IP          |  

### FaissIndexPQ  
Product Quantization  

**Параметры:**  
| Parameter | Type        | Domain          |  
|----------|-------------|-----------------|  
| M        | int         | Dividers of dim |  
| nbits    | int         | 1-10            |  
| metric   | Categorical | L2, IP          |  

### FaissIndexIVFScalarQuantizer  
Скалярная квантизация индексов с хранением в ячейках Вороного  

**Параметры:**  
| Parameter | Type        | Domain          |  
|----------|-------------|-----------------|  
| quantizer | Quantizer   | 8, 6, 4_bit, ...|  
| metric    | Categorical | L2, IP          |  
| nlist     | int         | 2-2?            |  

### FaissIndexIVFPQ  
Product Quantization с хранением в ячейках Вороного  

**Параметры:**  
| Parameter      | Type        | Domain          |  
|---------------|-------------|-----------------|  
| M             | int         | Dividers of dim |  
| nbits         | int         | 1-10            |  
| metric        | Categorical | L2, IP          |  
| nlist         | int         | 2-27?           |  
| M_refine      | int         | Dividers of dim |  
| nbits_refine  | int         | 1-10            |  

## TextNormalize  
Нормализация текста  

**Параметры:**  
| Parameter             | Type        | Domain               |  
|----------------------|-------------|----------------------|  
| stop_word_remove     | bool        | 1, 0                 |  
| word_generalization  | Categorical | stem, lemmatize, None|  

## Embedder  
Нормализация текста  

### BoWEmbedder  
На основе мешка слов  

### TfidfEmbedder  
На основе tf-idf  

### HuggingFaceEmbedder  

**Параметры:**  
| Parameter  | Type        | Domain                     |  
|-----------|-------------|----------------------------|  
| model_name| Categorical | sentence-transformers, ... |  

## Query  
Синхронизация запросов для полнотекстового поиска  

### RusVectoresQuery  

**Параметры:**  
| Parameter                   | Type   | Domain               |  
|----------------------------|--------|----------------------|  
| global_similarity_threshold | float  | 0-1                  |  
| POS_thresholds             | dict   | {PoS-tag: 0-1}       |  
| max_entries                | int    | 1 - chunks_amount    |  
| max_combinations           | int    | 1-22                 |  
| max_synonyms               | int    | 1-22?                |  

## Reranker  
Пересортировка результатов из разных выпадов по релевантности  

### LinearCombinationRerank  

**Параметры:**  
| Parameter               | Type  | Domain |  
|------------------------|-------|--------|  
| fulltext_score_weight  | float | 0-27?  |  
| vector_score_weight    | float | 0-22?  |  