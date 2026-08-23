import re
import math
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"[a-z0-9$§.%+-]+")

def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(x) for x in docs]
        self.n = len(self.docs)
        self.k1, self.b = k1, b
        self.lengths = np.array([max(1, len(x)) for x in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if self.n else 1.0
        self.tf = [Counter(x) for x in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.n - f + 0.5) / (f + 0.5))
            for t, f in df.items()
        }

    def scores(self, query):
        out = np.zeros(self.n, dtype=float)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                out[i] += idf * (freq * (self.k1 + 1)) / denom
        return out

def minmax(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.ptp(x) == 0:
        return np.zeros_like(x)
    return (x - x.min()) / np.ptp(x)

class HybridRetriever:
    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.bm25 = BM25([c["text"] for c in chunks])

    def retrieve(self, query, query_embedding, top_k=5):
        if not self.chunks:
            return []
        dense = cosine_similarity(
            np.asarray(query_embedding).reshape(1, -1), self.embeddings
        )[0]
        lexical = self.bm25.scores(query)
        score = 0.72 * minmax(dense) + 0.28 * minmax(lexical)
        order = np.argsort(-score)[:top_k]
        return [
            {**self.chunks[i],
             "score": float(score[i]),
             "dense_score": float(dense[i]),
             "lexical_score": float(lexical[i])}
            for i in order
        ]
