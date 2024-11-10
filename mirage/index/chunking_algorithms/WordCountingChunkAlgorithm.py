from .ChunkingAlgorithm import ChunkingAlgorithm


class WordCountingChunkingAlgorithm(ChunkingAlgorithm):

    def __init__(self, words = 50):
        self.words = words

    def __call__(self) -> dict[int, tuple[str, str]]:
        pass