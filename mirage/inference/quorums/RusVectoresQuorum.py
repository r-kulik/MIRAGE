import zipfile
import pymorphy3
from heapq import heappush, heappop
from gensim.models import KeyedVectors
from typing import List


class RusVectoresQuorum:
    def __init__(self, chunk_storage, path_to_model: str,
                 global_similarity_threshold: float = 0.75, 
                 second_closest_strategy: bool = True, POS_involved: List[str] = ['ADJS', 'ADJF', 'VERB'], 
                 max_entries: int = 50, max_combinations: int = 100):
        print("Я ДАУН")
        self.chunk_storage = chunk_storage
        self.global_similarity_threshold = global_similarity_threshold
        self.POS_involved = POS_involved
        self.max_entries = max_entries
        self.max_combinations = max_combinations
        self.morph = pymorphy3.MorphAnalyzer()
        self.word_vectors = KeyedVectors.load_word2vec_format(path_to_model, binary=True)

    def _get_pos(self, word: str) -> str:
        parsed = self.morph.parse(word)[0]
        return parsed.tag.POS

    def _inflect_synonym(self, synonym: str, original_word: str) -> str:
        parsed_original = self.morph.parse(original_word)[0]
        tags = parsed_original.tag
        parsed_syn = self.morph.parse(synonym)[0]
        inflected = parsed_syn.inflect(tags)
        return inflected.word if inflected else synonym

    def _get_synonyms(self, word: str, pos: str) -> List[tuple]:
        print('==================================')
        print(f"FIND SYNONIMS FOR WORD {word}")

        print(pos, type(pos))
        if pos not in self.POS_involved:
            return []
        parsed_word = self.morph.parse(word)[0]
        lemma = parsed_word.normal_form + '_' + str(pos)
        print(f'parsed lemma: {lemma}')
        print(f"does lemma belong to word vectors: {lemma in self.word_vectors}")
        if lemma not in self.word_vectors:
            return []
        try:
            synonyms = self.word_vectors.most_similar(lemma, topn=50)
        except KeyError:
            return []
        processed = []
        seen = set()
        for syn, sim in synonyms:
            if syn == lemma or sim < self.global_similarity_threshold:
                continue
            print(syn, word)
            inflected = self._inflect_synonym(syn.split('_')[0], word)
            if inflected not in seen:
                seen.add(inflected)
                processed.append((inflected, sim))
        print(f"processed synonims = {processed}")
        return processed

    def query(self, text: str) -> List:
        words = text.split()
        replaceable = []
        for idx, word in enumerate(words):
            pos = self._get_pos(word)
            syns = self._get_synonyms(word, pos)
            if syns:
                replaceable.append((idx, syns))

        heap = [(-0.0, 0.0, {})]
        seen = set()
        results = []
        combination_count = 0

        while heap and combination_count < self.max_combinations:
            current_priority, current_sim, replacements = heappop(heap)
            current_query = words.copy()
            for idx, syn_idx in replacements.items():
                for r_idx, (word_idx, syns) in enumerate(replaceable):
                    if word_idx == idx:
                        current_query[idx] = syns[syn_idx][0]
                        break
            modified = ' '.join(current_query)
            if modified in seen:
                continue
            seen.add(modified)
            print(modified)
            chunk_results = self.chunk_storage.query(modified)
            results.extend(chunk_results)
            if len(results) >= self.max_entries:
                return results[:self.max_entries]

            combination_count += 1
            if combination_count >= self.max_combinations:
                break

            for r_idx, (word_idx, syns) in enumerate(replaceable):
                if word_idx not in replacements:
                    for syn_idx, (syn, sim) in enumerate(syns):
                        new_replacements = replacements.copy()
                        new_replacements[word_idx] = syn_idx
                        new_sim = current_sim + sim
                        heappush(heap, (-new_sim, new_sim, new_replacements))
                else:
                    current_syn_idx = replacements[word_idx]
                    if current_syn_idx + 1 < len(syns):
                        new_replacements = replacements.copy()
                        new_replacements[word_idx] = current_syn_idx + 1
                        new_sim = current_sim - syns[current_syn_idx][1] + syns[current_syn_idx + 1][1]
                        heappush(heap, (-new_sim, new_sim, new_replacements))

        return results[:self.max_entries]