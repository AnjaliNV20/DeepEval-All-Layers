import re


class DocumentRetriever:

    def __init__(self, chunks: list[str]):
        self.chunks = chunks

    def _normalize(self, text: str) -> list[str]:

        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        stop_words = {
            "what",
            "which",
            "how",
            "does",
            "do",
            "is",
            "are",
            "the",
            "a",
            "an",
            "in",
            "for",
            "to",
            "of",
            "on",
            "can",
            "you"
        }

        return [
            word
            for word in text.split()
            if word not in stop_words
        ]

    def search(self, query: str) -> list[str]:

        query_words = set(self._normalize(query))

        if not query_words:
            return []

        scored_results = []

        for chunk in self.chunks:

            chunk_words = set(
                self._normalize(chunk)
            )

            matched_words = query_words & chunk_words

            if len(query_words) >= 3:

                match_ratio = (
                    len(matched_words)
                    / len(query_words)
                )

                # Lower threshold so valid
                # multi-word intent can be retrieved
                if match_ratio >= 0.50:

                    scored_results.append(
                        (match_ratio, chunk)
                    )

        # Highest relevance first
        scored_results.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            chunk
            for _, chunk in scored_results[:2]
        ]


class VectorRetriever:

    def __init__(
        self,
        vector_store,
        threshold: float = 0.60
    ):
        self.vector_store = vector_store
        self.threshold = threshold

    def search(self, query: str) -> list[str]:

        results = self.vector_store.search(
            query,
            top_k=5
        )

        relevant_results = []

        for result in results:

            distance = result["distance"]

            similarity = 1 / (1 + distance)

            if similarity >= self.threshold:

                relevant_results.append(
                    result["text"]
                )

        return relevant_results