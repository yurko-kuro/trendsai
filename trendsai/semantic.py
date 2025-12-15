# trendsai/semantic.py
from sentence_transformers import SentenceTransformer, util

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def semantic_rank(base: str, candidates: list[str]) -> list[tuple[str, float]]:
    """
    Повертає список (кандидат, cosine_score), відсортованих за схожістю.
    """
    if not candidates:
        return []

    model = get_model()
    emb_base = model.encode(base, convert_to_tensor=True)
    emb_all = model.encode(candidates, convert_to_tensor=True)

    scores = util.cos_sim(emb_base, emb_all)[0].cpu().tolist()
    pairs = list(zip(candidates, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs
