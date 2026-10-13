from sensai.core.llm.ollama import OllamaClient

_PREFIXES = {
    "qwen3-embedding": ("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:", ""),
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
    "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
}


class Embedder:
    def __init__(self, client: OllamaClient, model: str, max_chars: int, batch_size: int = 32):
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self._query_prefix, self._document_prefix = _PREFIXES.get(model.split(":")[0], ("", ""))
        prefix_length = max(len(self._query_prefix), len(self._document_prefix))
        self._options = {"num_ctx": max_chars + prefix_length}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors += self.client.embed(self.model, [self._document_prefix + text for text in batch], self._options)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed(self.model, [self._query_prefix + text], self._options)[0]
