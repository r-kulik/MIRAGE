import zipfile
# from loguru import logger
import pymorphy3
from heapq import heappush, heappop
from gensim.models import KeyedVectors
from typing import Dict, List, Self, Set, Tuple

from mirage.index import QueryResult
from mirage.index.chunk_storages.ChunkStorage import ChunkStorage, ChunkNote

# logger.disable(__name__)

class RusVectoresQuorum:
    """Synonimization module for enriching the search queries with the synonims
    """
    rus_vectores_POS_list: list[str] = ['ADJ', 'VERB', 'NUM', 'ADV', 'NOUN', 'INTJ', 'X']

    def __init__(self, chunk_storage: ChunkStorage, path_to_model: str,
                 global_similarity_threshold: float = 0.75, 
                 second_closest_strategy: bool = True, 
                 POS_thresholds: List[str] | Dict[str, float] = ['ADJ', 'VERB', 'NOUN', 'ADV'],
                 relative_cosine_similarity_strategy: bool = True,
                 max_entries: int = 50, max_combinations: int = 100,
                 max_synonims: int = 20,
                 visualize: bool = False) -> Self:
        """Initialization of the quroum and definition of its operation parameter

        Parameters
        ----------
        chunk_storage : ChunkStorage
            The fulltext-search storage that contains chunks of text
        path_to_model : str
            path to KeyedVectors Word2Vec model by RusVectores
        global_similarity_threshold : float, optional
            minimal value of cosine similarity between two words to consider them as a synonim, by default 0.75
        second_closest_strategy : bool, optional
            DEPRECATED, by default True
        POS_thresholds : List[str] | Dict[str, float], optional
            more accurate setting to set up a similarity treshold for different PoSes, for example:
            ```
            ...
            POS_thresholds=['ADJ']  # means only adjectives will be synonimized
            ...
            POS_thredholds={'NOUN': 0.89, 'VERB': 0.5, 'ADV': 0.92}  # means that NOUNS, VERBS and ADVERBS will be synonimized with the provided thresholds
            ...
            ````
        max_entries : int, optional
            Maximal amount of the chunks to be returned by default 50
        max_combinations : int, optional
            Maximal amount of the queries that will be send to ChunkStorage, by default 100
        max_synonims: int, optional
            How many synonims of the word will be considered for substitution
        visualize : bool, optional
            Must be true for printing the information about synonimization, by default False

        Raises
        ------
        ValueError
            Raises if POS_thresholds dict or list contains POS-tags that are not presented in the model 
        """
        self.chunk_storage = chunk_storage
        self.visualize = visualize
        
        # ---------------------------------------------------------------------
        # unification of the self.POS_threshold field. List, dict -> dict
        if type(POS_thresholds) == list:
            self.POS_thresholds = {x: global_similarity_threshold for x in POS_thresholds}
        else:
            self.POS_thresholds = POS_thresholds
        if not all(x in RusVectoresQuorum.rus_vectores_POS_list for x in self.POS_thresholds):
            raise ValueError('You have provided invalid POS tag in the POS thresholds argument')
        # ----------------------------------------------------------------------


        self.max_entries = max_entries
        self.max_combinations = max_combinations
        self.max_synonims = max_synonims
        self.morph = pymorphy3.MorphAnalyzer()

        # --------------------------------------------------
        # Strategy from the A Minimally Supervised Approach for Synonym Extraction with Word Embeddings
        self.relative_cosine_similarity_strategy: bool = relative_cosine_similarity_strategy
        self.TOP_N = 10
        self.RCS_THRESHOLD = 0.11
        # ------------ loading the Word2Vec Model from the file
        self.word_vectors: KeyedVectors = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
        
        if self.visualize: 
            print(f'Amount of vectors loaded in quorum: {len(self.word_vectors)}')
            print(f'word "быть" is presented in word vectors: {"быть_VERB" in self.word_vectors}')
            print(f'set of POS in w2v model: {set([x.split("_")[1] for x in self.word_vectors.key_to_index])}')
            print(f'Rules of synonimization: {self.POS_thresholds}')

    @staticmethod
    def downcast_pos(pymorphy_tag: str) -> str:
        """Function for the translation of pymorphy tags in the RusVectores tag set

        Parameters
        ----------
        pymorphy_tag : str
            Tag obtained from the pymorphy

        Returns
        -------
        str
            Tag in the set of RusVectores tags: 
            ```
['ADJ', 'VERB', 'NUM', 'ADV', 'NOUN', 'INTJ', 'X']
            ```
        """
        if pymorphy_tag in {'ADJF', 'ADJS'}:
            return 'ADJ'
        elif pymorphy_tag in {'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND'}:
            return 'VERB'
        elif pymorphy_tag == 'NUMR':
            return 'NUM'
        elif pymorphy_tag == 'ADVB':
            return 'ADV'
        elif pymorphy_tag in {'NOUN', 'NPRO', 'PROPN'}:
            return 'NOUN'
        elif pymorphy_tag == 'INTJ':
            return 'INTJ'
        else:
            return 'X'  # Other unknown tags will be considered as X tag

    def _get_pos(self, word: str) -> str:
        """Obtaining PartOfSpeech tag of the word from the pymorphy

        Parameters
        ----------
        word : str

        Returns
        -------
        str
            PoS of the word
        """     
        parsed = self.morph.parse(word)[0]
        return parsed.tag.POS

    def _inflect_synonym(self, synonym: str, original_word: str) -> str:
        """Performs the inflection (склонение, изменение формы слова)

        Parameters
        ----------
        synonym : str
            word needed to be inflected in the form of the original word
        original_word : str

        Returns
        -------
        str
            inflected in the form of the original word synonim
        """
        # -----------------------------------------------------
        # Getting the tagset of the original word. It contains all grammar
        # information about its form needed to inflect a synonim
        parsed_original = self.morph.parse(original_word)[0]
        tags = set(parsed_original.tag._grammemes_tuple) # do not change
        # ------------------------------------------------------
        # performing inflection of the synonim with the obtained tagset
        parsed_syn = self.morph.parse(synonym)[0]
        inflected = parsed_syn.inflect(tags)
        # ------------------------------------------------------
        return inflected.word if inflected else synonym
    
    def get_relative_cosine_similarity_synonyms(self, lemma: str):
        # logger.debug(f"Getting relative cosine similarity synonyms for {lemma}")
        try:
            candidate_synonyms = self.word_vectors.most_similar(
                positive=lemma, topn=self.TOP_N
            )
        except KeyError:
            return []
        # logger.debug(f"Candidate synonyms for {lemma}: {candidate_synonyms}")
        total_similarity = sum(sim for _, sim in candidate_synonyms)
        filtered_synonyms = [
            (syn, sim) for syn, sim in candidate_synonyms
            if sim / total_similarity > self.RCS_THRESHOLD
        ]
        return filtered_synonyms 

    def _get_synonyms(self, word: str, pos: str) -> List[Tuple[str, float]]:
        """Returns the list of the synonims of the provided form

        Parameters
        ----------
        word : str
        pos : str
            PoS tag of the word obtained from the pymorphy

        Returns
        -------
        List[Tuple[str, float]]
            List of pairs (synonim, cosine similarity) of the word
            
        """
        # logger.debug(f"Getting synonyms for {word} with pos {pos}")
        # --------------------------------------------------------
        # downcasting pymorphy PoS-tag to RusVectores PoS-tag
        downcasted_pos = RusVectoresQuorum.downcast_pos(pos)
        # -------------------------------------------------------- 
        # we are not looking up  synonims for PoS'es not presented in the PoS_thresholds 
        # logger.debug(f"Downcasted pos: {downcasted_pos}")
        if downcasted_pos not in self.POS_thresholds:
            return []
        # logger.debug('')
        # --------------------------------------------------------
        # Obtaining lemmatized (dictionary) form of the word
        parsed_word = self.morph.parse(word)[0]
        lemma = parsed_word.normal_form + '_' + downcasted_pos
        # --------------------------------------------------------
        # lemma is not known for the RusVectotes model, return empty list
        # logger.debug(f"Obtained lemma {lemma} for {word}")
        if lemma not in self.word_vectors:
            # logger.warning(f"Word {lemma} is not in the model")
            return []
        # --------------------------------------------------------
        # trying to obtain max_synonims close words
        if self.relative_cosine_similarity_strategy:
            synonyms = self.get_relative_cosine_similarity_synonyms(
                lemma
            )
        else:
            try:
                synonyms = self.word_vectors.most_similar(lemma, topn=self.max_synonims)
            except KeyError:
                return []
            # --------------------------------------------------------
        
        if self.visualize: 
            print(f" For word \"{word}\" obtained synonims:\n{synonyms}")
        
        # --------------------------------------------------------
        # creating list of inflected (turned in the form of original word) synonims
        processed = []
        seen = set()
        # --------------------------------------------------------
        # in the cycle for each pair of the synonim-similarity_score:
        for syn, sim in synonyms:
            # --------------------------------------------------------
            # Skip tokens with the same lemma (but possibly different pos)
            # or with the lower then needed similarity
            if syn == lemma or sim < self.POS_thresholds[downcasted_pos]:
                continue
            # --------------------------------------------------------
            # obtaining pure word from token and trying (!) to inflect
            # if it is not possible to inflect the synonim in the same way
            # as word (because of gramatical reasons), the synonims stays unchanged
            syn = syn.split('_')[0]
            inflected = self._inflect_synonym(syn, word)
            # --------------------------------------------------------
            # taking only unique synonims in the consideration
            if inflected not in seen:
                seen.add(inflected)
                processed.append((inflected, sim))
            # --------------------------------------------------------
        # --------------------------------------------------------
        return processed

    def query(self, text: str) -> List[ChunkNote]:
        """
        Querying ChunkStorage with the closest possible synonimized queries
        Parameters
        ----------
        text : str
            query text

        Returns
        -------
        List[ChunkNote]
            Search results (need to be reweighted)
        """
        # --------------------------------------------------------
        # Creating the list of words and substitutions (replaceable)
        # that contains note like this:
        # (1, [('обожать', 0.75), ('восхищаться', 0.69), ...])
        # that means that the word with index 1 in the original query can
        # be replaced with the following synonims considering the procided cosine similarity
        words: List[str] = text.split()
        replaceable: list[Tuple[int, List[Tuple[str, float]]]] = []
        # --------------------------------------------------------
        # in cycle for each index and its word filling the replaceable array
        for idx, word in enumerate(words):
            pos = self._get_pos(word)
            syns = self._get_synonyms(word, pos)
            # --------------------------------------------------------
            # only if there are some synonims, words without synonims will not be changed
            if syns:
                replaceable.append((idx, syns))
            # --------------------------------------------------------
        # --------------------------------------------------------

        # --------------------------------------------------------
        # Creating a heap with the following structure:
        # (priority, similarity, replacements dict ({word_index: synonim_index}) )
        heap = [(-1, 1, {})]
        # --------------------------------------------------------
        # Creating a set of the seen combinations of substitutions
        # and results obtained from the chunk storage
        seen = set()
        results: Set[QueryResult] = set()
        # --------------------------------------------------------
        combination_count = 0
        # --------------------------------------------------------
        # while heap is not empty and we have not reached the maximal amount of combinations
        while heap and combination_count < self.max_combinations:
            # --------------------------------------------------------
            # we are popping the element with the highest priority (lowest current sim)
            current_priority, current_sim, replacements = heappop(heap)
            current_query = words.copy()
            # --------------------------------------------------------
            # creating modified query (list of words in query) by applying 
            # substitutions from the replacements dict
            # idx is an index of word, 
            # syn_indx is and index of the  synonim in the list of the synonims
            for idx, syn_idx in replacements.items():
                # --------------------------------------------------------
                # iterating through pairs in replaceable array
                # (word_indx, syns = [(syn_1, sim_1), (syn_2, sim_2), ...]
                for r_idx, (word_idx, syns) in enumerate(replaceable):
                    # --------------------------------------------------------
                    # if word_index is an index presented in the replacements
                    if word_idx == idx:
                        #--------------------------------------------------------
                        # we are changing the word in current query to its synonim
                        current_query[idx] = syns[syn_idx][0] 
                        # and finising the search of synonims for this word
                        break
                        # --------------------------------------------------------
                    #--------------------------------------------------------
                # --------------------------------------------------------
            # --------------------------------------------------------
            # joining mofigied query list of word in the unique text
            modified = ' '.join(current_query)
            if modified in seen:
                continue
            seen.add(modified)
            # --------------------------------------------------------
            # quering the chunk storage with this query
            if self.visualize: print(f"SEARCH QUERY with sim {current_sim}: {modified}")
            chunk_results = self.chunk_storage.query(modified)
            # --------------------------------------------------------
            # adding the results of this query in the set of the results
            # without overfitting it
            for chunk_result in chunk_results:
                chunk_result.score *= current_sim
                if chunk_result not in results:
                    results.add(chunk_result)
                if len(results) >= self.max_entries:
                    return list(sorted(results, key=lambda x: x.score, reverse=True))
            
            # --------------------------------------------------------
            # or stopping the cycle if we have reached the maximal amount of iterations
            combination_count += 1
            if combination_count >= self.max_combinations:
                break
            # --------------------------------------------------------
            # adding new elements in the heap. How?
            # for every word that can be replaced:
            for r_idx, (word_idx, syns) in enumerate(replaceable):
                # --------------------------------------------------------
                # adding all replacements that perform only ONE change 
                # from the current state of the replacements dict
                # ONLY if this word was not change in the current iteration of the algorithm
                if word_idx not in replacements:
                    for syn_idx, (syn, sim) in enumerate(syns):
                        new_replacements = replacements.copy()
                        new_replacements[word_idx] = syn_idx
                        new_sim = current_sim * sim
                        heappush(heap, (-new_sim, new_sim, new_replacements))
                # --------------------------------------------------------
                # but if the word WAS changed, we are not adding its substiturion, but trying
                # to push a new synonim in the heap instead of the current replacement if
                # it is possible
                else:
                    current_syn_idx = replacements[word_idx]
                    if current_syn_idx + 1 < len(syns):
                        new_replacements = replacements.copy()
                        new_replacements[word_idx] = current_syn_idx + 1
                        new_sim = current_sim / syns[current_syn_idx][1] * syns[current_syn_idx + 1][1]
                        heappush(heap, (-new_sim, new_sim, new_replacements))
                # --------------------------------------------------------
            # and running a new cycle with new heap
        # --------------------------------------------------------
        # returning the list of the results in the end
        return list(sorted(results, key=lambda x: x.score, reverse=True ))
    #--------------------------------------------------------