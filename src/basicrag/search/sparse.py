"""Lexical retrieval with a dependency-free BM25 implementation.

BM25 (Best Matching 25) is a ranking function used by search engines to score
how relevant a document is to a query based on term frequency. The pipeline is:

1. Tokenize documents and split them into terms.
2. Build an inverted index (term -> list of (document, frequency)).
3. Compute IDF (inverse document frequency) for every term: rare terms weigh more.
4. For a query, sum the per-term BM25 contributions for each candidate document.
5. Normalize by document length and rank the documents by score.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    # Step 1: tokenization. Lowercase the text and extract word characters
    # (letters, digits, underscore) so that "Cat" and "cat" match the same term.
    return _TOKEN.findall(text.lower())


def matches_filter(metadata: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if callable(expected):
            if not expected(actual):
                return False
        elif isinstance(expected, (list, tuple, set, frozenset)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _with_score(document: Document, score: float) -> Document:
    # Attach the computed BM25 score to the document metadata so callers can
    # inspect ranking results without a separate return type.
    metadata = {**(document.metadata or {}), "bm25_score": round(float(score), 6)}
    return document.model_copy(update={"metadata": metadata})


class BM25Index:
    """Inverted-index BM25 over a fixed set of documents."""

    def __init__(
        self,
        documents: list[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        # k1 controls term-frequency saturation: higher values let repeated terms
        # keep contributing for longer; b controls length normalization (0 = none,
        # 1 = fully normalize against the average document length).
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths: list[int] = []
        # Step 2: inverted index. postings[term] = [(document_index, term_count), ...]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}

        # document_frequency[term] = number of documents containing that term.
        document_frequency: Counter[str] = Counter()
        for index, document in enumerate(documents):
            tokens = tokenize(document.page_content or "")
            frequencies = Counter(tokens)  # term frequency within this document
            self.doc_lengths.append(len(tokens))
            for term, frequency in frequencies.items():
                self.postings[term].append((index, frequency))
                document_frequency[term] += 1

        # Average document length (avgdl) is used to normalize scores; fall back
        # to 1.0 when the corpus is empty to avoid division by zero.
        count = max(1, len(documents))
        self.average_length = (sum(self.doc_lengths) / count if self.doc_lengths else 0.0) or 1.0

        # Step 3: IDF via the BM25 "probabilistic" formula (the +0.5 smoothing
        # avoids negative/zero scores and the log grows as a term becomes rarer).
        #   idf = ln(1 + (N - n + 0.5) / (n + 0.5))
        # where N = total documents, n = documents containing the term.
        for term, frequency in document_frequency.items():
            self.idf[term] = math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))

    @classmethod
    def from_documents(
        cls, documents: list[Document], *, k1: float = 1.5, b: float = 0.75
    ) -> BM25Index:
        return cls(documents, k1=k1, b=b)

    def search(
        self,
        query: str,
        k: int = 4,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        # Step 4: tokenize the query and deduplicate terms (a repeated query term
        # should not multiply the score; BM25 already uses per-document frequency).
        terms = set(tokenize(query))
        if not terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.average_length
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue  # query term never appeared in the corpus -> no contribution

            # Only documents listed in the postings for this term can score on it.
            for index, frequency in self.postings[term]:
                document = self.documents[index]
                if not matches_filter(document.metadata or {}, filters):
                    continue

                # Step 5: BM25 term score:
                #   idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                # - tf: frequency of the term in this document
                # - dl: document length, avgdl: average document length
                # - higher tf -> higher score, but saturates via k1
                # - longer documents are penalized via b
                length = self.doc_lengths[index] or 1
                denominator = frequency + k1 * (1.0 - b + b * length / avgdl)
                scores[index] += idf * frequency * (k1 + 1.0) / denominator

        # Step 6: rank documents by total BM25 score (descending) and keep top-k.
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [(self.documents[index], score) for index, score in ranked]

    def __len__(self) -> int:
        return len(self.documents)


class BM25Retriever(BaseRetriever):
    """LangChain retriever backed by :class:`BM25Index`."""

    index: BM25Index
    k: int = 4
    filter: dict[str, Any] | None = Field(default=None)

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        *,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Retriever:
        return cls(
            index=BM25Index.from_documents(documents, k1=k1, b=b),
            k=k,
            filter=filter,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        return [
            _with_score(document, score)
            for document, score in self.index.search(query, self.k, self.filter)
        ]


def bm25_search(
    documents: Sequence[Document],
    query: str,
    *,
    k: int = 4,
    filter: dict[str, Any] | None = None,
) -> list[Document]:
    return BM25Retriever.from_documents(list(documents), k=k, filter=filter).invoke(query)
