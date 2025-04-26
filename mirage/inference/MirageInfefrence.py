
from abc import ABC

from mirage.inference.llm_adapters.LLMAdapter import LLMAdapter
from mirage.inference.quorums.RusVectoresQuorum import RusVectoresQuorum
from mirage.inference.rerankers.Reranker import Reranker


class MirageInference:


    def __init__(
        self,
        llm_adapter: LLMAdapter,
        quorum: RusVectoresQuorum,
        reranker: Reranker
    ):
        self.llm_adapter = llm_adapter
        self.quorum = quorum
        self.reranker = reranker
    
    def save():
        ...

    def load():
        ...