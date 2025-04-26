from typing import List
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from abc import ABC, abstractmethod
import numpy as np

class RelevanceFunction(ABC):
    def __init__(self, threshhold: float = 0.8):
        self.threshhold = threshhold
        self.embedder = None
        self.ideal_context = None
        
    def set_embedder(self, embedder):
        self.embedder = embedder
        
    @abstractmethod
    def set_ideal_context(self, ideal_context):
        pass
        
    @abstractmethod
    def get_relevance_float(self, str_model: str) -> float:
        pass
    
    def get_relevance(self, str_model: str) -> List[bool]:
        score = self.get_relevance_float(str_model)
        return (score, int(score > self.threshhold))
    
class CosineRelevance(RelevanceFunction):
    def __init__(self, threshhold = 0.8):
        super().__init__(threshhold)
        
    def set_ideal_context(self, ideal_context):
        self.ideal_context = self.embedder.embed(text=ideal_context).reshape(1, -1)
    
    def get_relevance_float(self, str_model):
        v = self.embedder.embed(text=str_model).reshape(1, -1)
        
        return cosine_similarity(self.ideal_context, v)[0][0]
    
class IoU(RelevanceFunction):
    def __init__(self, threshhold = 0.8):
        super().__init__(threshhold)
        
    def set_ideal_context(self, ideal_context):
        self.ideal_context = set(ideal_context.split())

    def get_relevance_float(self, str_model):
        arr_model = str_model.split()
        
        res = len(set(arr_model) & self.ideal_context) / len(arr_model)
        
        return res
    

class ContextPrecision:
    ...

class ContextRecall:
    ...