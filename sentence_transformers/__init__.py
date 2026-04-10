"""
Stub sentence_transformers package for testing without the real library.
The real SentenceTransformer is mocked in tests via unittest.mock.patch.
"""


class SentenceTransformer:
    """Stub — replaced by mock in tests via patch('sentence_transformers.SentenceTransformer')."""

    def __init__(self, model_name_or_path: str, **kwargs):
        self.model_name_or_path = model_name_or_path
        self._dim = 384

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, sentences, **kwargs):
        n = len(sentences) if isinstance(sentences, list) else 1
        return [[0.0] * self._dim for _ in range(n)]
