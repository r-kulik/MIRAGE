from collections import namedtuple
from typing import List, Tuple
from loguru import logger
import numpy as np

ScorePairType = namedtuple("ScorePairType", ["score", "relevance"])


class RankMetr:
    @staticmethod
    def ndcg(rel_scores: List[ScorePairType]):
        rel_scores = [i[1] for i in rel_scores]
        dcg_score = 0
        for i, rel in enumerate(rel_scores, start=1):
            dcg_score += rel / np.log2(i + 1)
        if dcg_score == 0:
            logger.info(rel_scores)

        idcg_score = 0
        for i, rel in enumerate(sorted(rel_scores), start=1):
            idcg_score += 1 / np.log2(i + 1)
        if idcg_score == 0:
            logger.eror(f"dcg = {dcg_score}")
            raise ValueError
        return dcg_score / idcg_score if idcg_score != 0 else 0.0

    @staticmethod
    def ap(rel_scores):

        rel_scores = [i[1] for i in rel_scores]

        # Ensure k is not larger than the list
        k = len(rel_scores)

        # Count the number of relevant items
        relevant_items = sum(rel_scores[:k])

        if relevant_items == 0:
            return 0.0  # If there are no relevant items, AP@k is 0

        # Compute precision at each position where a relevant item is found
        ap = 0.0
        cumulative_relevant = 0

        for i in range(k):
            if rel_scores[i] == 1:
                cumulative_relevant += 1
                precision_at_i = cumulative_relevant / (i + 1)
                ap += precision_at_i

        # Normalize by the total number of relevant items in the top k
        ap /= relevant_items

        return ap

    @staticmethod
    def erp(relevance_scores):

        relevance_scores = [i[0] for i in relevance_scores]

        # Ensure k is not larger than the list
        k = len(relevance_scores)

        err = 0.0
        product = 1.0  # Represents the product of (1 - r_j) for j < i

        for i in range(k):
            r_i = relevance_scores[i]
            if r_i < 0 or r_i > 1:
                raise ValueError("Relevance scores must be between 0 and 1.")

            # Probability of stopping at position i
            p_i = product * r_i

            # Add to ERR
            err += p_i / (i + 1)

            # Update the product for the next iteration
            product *= 1 - r_i

        return err
