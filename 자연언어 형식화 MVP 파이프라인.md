---
생성일시:
  - 2026-01-29 17:42
---
# 골격 코드
```
"""
Self-Referential Natural-Language Formalization (MVP skeleton)

We formalize a concept C into operators:
  x_{t+1} = A x_t + b
  W_{t+1} = (1-eta) W_t + eta * S(x_t), with pruning

- x_t: d-dim embedding vector
- W_t: NxN weighted relation matrix over concept-nodes (prototypes u_j)

This is a "minimum math form" implementation skeleton.
You must provide:
  1) embedding_model.encode(text) -> np.ndarray shape (d,)
  2) corpus: list[str] (sentences/short paragraphs)
  3) seed contexts for each concept-node (or a method to collect them)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np


# -----------------------------
# Utils
# -----------------------------

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, eps)

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(a, b) / (max(np.linalg.norm(a), eps) * max(np.linalg.norm(b), eps)))

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def spectral_norm(A: np.ndarray) -> float:
    # largest singular value
    # For MVP: use full SVD. For large d, switch to power iteration.
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0])

def ridge_closed_form(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve B = argmin ||B X - Y||_F^2 + lam ||B||_F^2
    X: (d+1, m)
    Y: (d,   m)
    returns B: (d, d+1)
    """
    d1, m = X.shape
    I = np.eye(d1)
    # B = Y X^T (X X^T + lam I)^-1
    XXT = X @ X.T
    inv = np.linalg.inv(XXT + lam * I)
    B = (Y @ X.T) @ inv
    return B


# -----------------------------
# Embedding Model Interface
# -----------------------------

class EmbeddingModel:
    def encode(self, text: str) -> np.ndarray:
        """
        Return a 1D numpy array of shape (d,).
        Replace this with your actual embedding model.
        """
        raise NotImplementedError


# -----------------------------
# Core Config
# -----------------------------

@dataclass
class SRConfig:
    # data construction
    k_contexts: int = 1000      # sample size for concept C contexts
    k_neighbors: int = 8        # number of "next context" candidates per context

    # ridge + contraction
    ridge_lambda: float = 1e-2
    contraction_eps: float = 1e-3  # enforce ||A||_2 <= 1 - eps

    # upper-layer
    eta: float = 0.15
    beta: float = 10.0           # softmax temperature for a(x)
    prune_theta: float = 1e-3    # pruning threshold

    # rollout
    t_max: int = 50
    eps_x: float = 1e-4
    eps_w: float = 1e-4

    # reproducibility
    seed: int = 7


# -----------------------------
# Context collection (placeholders)
# -----------------------------

def sample_contexts_for_concept(
    concept: str,
    corpus: Sequence[str],
    n: int,
    rng: np.random.Generator,
) -> List[str]:
    """
    MVP heuristic: pick sentences containing the concept token.
    Replace with your own retrieval (BM25, regex, semantic search, etc.).
    """
    hits = [s for s in corpus if concept in s]
    if not hits:
        # fallback: random sample
        hits = list(corpus)

    if len(hits) <= n:
        return hits
    idx = rng.choice(len(hits), size=n, replace=False)
    return [hits[i] for i in idx]

def get_next_sentences(
    ctx: str,
    corpus: Sequence[str],
    window: int = 2
) -> List[str]:
    """
    MVP: treat corpus as ordered list; find occurrences of ctx and take following items.
    If duplicates, returns first match.
    Replace with document-aware logic.
    """
    out: List[str] = []
    try:
        i = corpus.index(ctx)
    except ValueError:
        return out

    for j in range(1, window + 1):
        if i + j < len(corpus):
            out.append(corpus[i + j])
    return out

def single_edit_paraphrases(ctx: str) -> List[str]:
    """
    MVP placeholder: returns empty.
    In a real system, generate single-step paraphrases or rule-based transforms.
    """
    return []

def propose_next_contexts(
    ctx: str,
    corpus: Sequence[str],
    k_neighbors: int,
) -> List[str]:
    """
    Minimal 'meaning-generation' proxy:
      - next sentences in corpus
      - plus optional paraphrases (single edit)
    """
    candidates: List[str] = []
    candidates += get_next_sentences(ctx, corpus, window=3)
    candidates += single_edit_paraphrases(ctx)
    # dedupe, keep order
    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen and c.strip():
            seen.add(c)
            deduped.append(c)
    return deduped[:k_neighbors]


# -----------------------------
# Prototypes (concept nodes) u_j
# -----------------------------

def build_prototypes(
    concept_nodes: Sequence[str],
    corpus: Sequence[str],
    embed: EmbeddingModel,
    cfg: SRConfig,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build u_j for each concept node (prototype vector).
    Simplest: centroid of embeddings of contexts containing that node string.

    Returns:
      U: (N, d)
      node_index: mapping node -> j
    """
    rng = np.random.default_rng(cfg.seed)
    node_index = {node: j for j, node in enumerate(concept_nodes)}
    U_list: List[np.ndarray] = []

    for node in concept_nodes:
        ctxs = sample_contexts_for_concept(node, corpus, n=min(cfg.k_contexts, 200), rng=rng)
        vecs = [embed.encode(c) for c in ctxs]
        if not vecs:
            raise ValueError(f"No contexts found for node: {node}")
        u = np.mean(np.stack(vecs, axis=0), axis=0)
        U_list.append(l2_normalize(u))

    U = np.stack(U_list, axis=0)
    return U, node_index


# -----------------------------
# Lower operator f: x_{t+1} = A x_t + b
# -----------------------------

@dataclass
class LowerOperator:
    A: np.ndarray  # (d, d)
    b: np.ndarray  # (d,)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x + self.b


def fit_lower_operator(
    concept: str,
    corpus: Sequence[str],
    embed: EmbeddingModel,
    cfg: SRConfig,
) -> LowerOperator:
    """
    Build transitions:
      x = E(ctx)
      y = mean(E(next_ctx)) over proposed next contexts
    Fit ridge to get [A b] with closed form, then enforce contraction on A.
    """
    rng = np.random.default_rng(cfg.seed)
    ctxs = sample_contexts_for_concept(concept, corpus, n=cfg.k_contexts, rng=rng)

    X_cols: List[np.ndarray] = []
    Y_cols: List[np.ndarray] = []

    for ctx in ctxs:
        next_ctxs = propose_next_contexts(ctx, corpus, k_neighbors=cfg.k_neighbors)
        if not next_ctxs:
            continue

        x = embed.encode(ctx)
        nxt_vecs = [embed.encode(nc) for nc in next_ctxs]
        y = np.mean(np.stack(nxt_vecs, axis=0), axis=0)

        X_cols.append(x)
        Y_cols.append(y)

    if not X_cols:
        raise ValueError("No transition pairs created. Check corpus/proposal rules.")

    X = np.stack(X_cols, axis=1)  # (d, m)
    Y = np.stack(Y_cols, axis=1)  # (d, m)
    d, m = X.shape

    # augment for bias
    ones = np.ones((1, m))
    Xp = np.vstack([X, ones])     # (d+1, m)

    B = ridge_closed_form(Xp, Y, lam=cfg.ridge_lambda)  # (d, d+1)
    A = B[:, :d]
    b = B[:, d]

    # enforce contraction: ||A||_2 <= 1 - eps
    sn = spectral_norm(A)
    target = 1.0 - cfg.contraction_eps
    if sn > target:
        A = (target / sn) * A

    return LowerOperator(A=A, b=b)


# -----------------------------
# Upper operator Φ: W_{t+1} = (1-eta)W_t + eta*S(x_t)
# S(x) = a(x) a(x)^T with a_j(x) = softmax(beta*cos(x,u_j))
# -----------------------------

@dataclass
class UpperOperator:
    U: np.ndarray        # (N, d) prototypes
    eta: float
    beta: float
    prune_theta: float

    def a(self, x: np.ndarray) -> np.ndarray:
        # similarities to prototypes
        x_n = l2_normalize(x)
        sims = self.U @ x_n  # (N,) because U rows are already normalized
        return softmax(self.beta * sims)

    def S(self, x: np.ndarray) -> np.ndarray:
        a = self.a(x)  # (N,)
        return np.outer(a, a)  # (N,N)

    def __call__(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        W_new = (1.0 - self.eta) * W + self.eta * self.S(x)
        # prune
        W_new = np.where(W_new >= self.prune_theta, W_new, 0.0)
        return W_new


# -----------------------------
# Rollout & convergence
# -----------------------------

@dataclass
class RolloutResult:
    converged: bool
    steps: int
    x_star: np.ndarray
    W_star: np.ndarray
    history: Dict[str, List[float]]

def rollout(
    f: LowerOperator,
    Phi: UpperOperator,
    x0: np.ndarray,
    W0: np.ndarray,
    cfg: SRConfig,
) -> RolloutResult:
    x = x0.copy()
    W = W0.copy()

    hist_dx: List[float] = []
    hist_dW: List[float] = []
    hist_mass: List[float] = []

    for t in range(1, cfg.t_max + 1):
        x_new = f(x)
        W_new = Phi(W, x)

        dx = float(np.linalg.norm(x_new - x))
        dW = float(np.linalg.norm(W_new - W, ord="fro"))
        mass = float(np.sum(W_new))

        hist_dx.append(dx)
        hist_dW.append(dW)
        hist_mass.append(mass)

        if dx <= cfg.eps_x and dW <= cfg.eps_w:
            return RolloutResult(
                converged=True,
                steps=t,
                x_star=x_new,
                W_star=W_new,
                history={"dx": hist_dx, "dW": hist_dW, "mass": hist_mass},
            )

        x, W = x_new, W_new

    return RolloutResult(
        converged=False,
        steps=cfg.t_max,
        x_star=x,
        W_star=W,
        history={"dx": hist_dx, "dW": hist_dW, "mass": hist_mass},
    )


# -----------------------------
# Main: Formalize a concept C
# -----------------------------

@dataclass
class Formalization:
    concept: str
    lower: LowerOperator
    upper: UpperOperator
    x0: np.ndarray
    W0: np.ndarray
    result: RolloutResult
    node_index: Dict[str, int]


def formalize_concept_mvp(
    concept: str,
    corpus: Sequence[str],
    embed: EmbeddingModel,
    concept_nodes: Sequence[str],
    cfg: Optional[SRConfig] = None,
) -> Formalization:
    """
    Minimum end-to-end:
      1) build prototypes U for concept_nodes
      2) fit lower operator (A,b) for concept
      3) set x0 = centroid of concept contexts
      4) set W0 = zeros
      5) rollout until convergence

    concept_nodes: small "dictionary" of nodes (including the target concept if you like)
    """
    cfg = cfg or SRConfig()
    rng = np.random.default_rng(cfg.seed)

    # Prototypes U (N,d)
    U, node_index = build_prototypes(concept_nodes, corpus, embed, cfg)

    # Fit f
    lower = fit_lower_operator(concept, corpus, embed, cfg)

    # x0 = centroid of contexts for concept
    ctxs = sample_contexts_for_concept(concept, corpus, n=min(cfg.k_contexts, 300), rng=rng)
    X0 = np.mean(np.stack([embed.encode(c) for c in ctxs], axis=0), axis=0)
    x0 = X0

    # W0 = zeros
    N = U.shape[0]
    W0 = np.zeros((N, N), dtype=float)

    # Phi
    upper = UpperOperator(U=U, eta=cfg.eta, beta=cfg.beta, prune_theta=cfg.prune_theta)

    # rollout
    result = rollout(lower, upper, x0, W0, cfg)

    return Formalization(
        concept=concept,
        lower=lower,
        upper=upper,
        x0=x0,
        W0=W0,
        result=result,
        node_index=node_index,
    )


# -----------------------------
# Example Usage (You must implement EmbeddingModel)
# -----------------------------
if __name__ == "__main__":
    class DummyEmbed(EmbeddingModel):
        def __init__(self, d: int = 16):
            self.d = d
            self.rng = np.random.default_rng(0)

        def encode(self, text: str) -> np.ndarray:
            # WARNING: dummy; replace with real embeddings
            h = abs(hash(text)) % (10**9)
            r = np.random.default_rng(h)
            v = r.normal(size=(self.d,))
            return v.astype(float)

    # Example corpus (replace with your own)
    corpus = [
        "규범성은 개념 간 관계를 정의한다.",
        "본질은 의미공간에서 응축되는 지점이다.",
        "자기지시 루프는 의미 생성 규칙을 반복한다.",
        "따라서 자연언어 형식화는 연산으로 전환된다.",
        "하지만 개념은 다의성이 있으므로 분해가 필요할 수 있다.",
        "즉 의미 분포의 중심부를 잡아야 한다.",
    ]

    embed = DummyEmbed(d=32)
    cfg = SRConfig(k_contexts=200, k_neighbors=2, t_max=30)

    # concept nodes (small dictionary)
    nodes = ["규범성", "본질", "자기지시", "의미공간", "형식화"]

    F = formalize_concept_mvp(
        concept="본질",
        corpus=corpus,
        embed=embed,
        concept_nodes=nodes,
        cfg=cfg,
    )

    print("Converged:", F.result.converged, "steps:", F.result.steps)
    print("dx last:", F.result.history["dx"][-1], "dW last:", F.result.history["dW"][-1])
    print("A spectral norm:", spectral_norm(F.lower.A))
    print("W mass:", np.sum(F.result.W_star))

```

# 추가 모듈
좋습니다. “다음 단계”는 제가 앞에서 말한 그대로 **두 군데를 실전형으로 바꾸는 것**입니다.

1. `EmbeddingModel.encode()`를 **실제 임베딩**으로 교체
    
2. `propose_next_contexts()`를 **문서/담화 구조 + 접속어 규칙 + 1-step 변형**을 반영하도록 강화
    

아래는 **바로 붙여 넣어서** 기존 골격 코드에 연결할 수 있는 형태로 드립니다.

---

## 1) 실제 임베딩으로 교체 (로컬 / API 두 옵션)

### 옵션 A: 로컬(sentence-transformers) — 제일 간단, 오프라인 가능

```python
# pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

class STEmbed(EmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        v = self.model.encode(text, normalize_embeddings=False)
        return np.asarray(v, dtype=float)
```

- 장점: 구현이 끝, 로컬에서 바로 실험 가능
    
- 단점: 모델 선택에 따라 한국어 품질 편차가 있습니다
    

---

### 옵션 B: OpenAI 임베딩 API — 품질 안정, 인터넷 필요

여기는 사용 환경에 따라 API키/네트워크가 필요해서, 코드 형태만 드리겠습니다.

```python
# pip install -U openai
import numpy as np
from openai import OpenAI

class OpenAIEmbed(EmbeddingModel):
    def __init__(self, model: str = "text-embedding-3-large", api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def encode(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=text)
        v = resp.data[0].embedding
        return np.asarray(v, dtype=float)
```

---

## 2) `propose_next_contexts()` 강화 (실전형)

핵심은 3갈래로 후보를 만들고, **벡터 유사도로 랭킹**해서 top-k만 쓰는 겁니다.

- (i) **문서/담화 연쇄**: 같은 문서(혹은 같은 스레드)에서 다음 1~3문장
    
- (ii) **마커 기반 분기**: “따라서/즉/하지만/다만/반면/구체적으로…” 뒤 문장
    
- (iii) **1-step 변형**: 단일 규칙 변형(최소 MVP에서는 “접속어를 붙여 확장”만 해도 효과 큼)
    

### 2-1) 먼저, 코퍼스를 “문서 단위”로 다루는 구조로 바꾸는 게 좋습니다

지금 골격은 `corpus: list[str]` 한 줄 배열이었는데, 실전에서는 문서가 섞이면 노이즈가 커집니다.  
그래서 최소로 이렇게 바꾸세요.

```python
# corpus_docs: list[list[str]]
#   - 각 문서가 문장 리스트
#   - 대화면 "대화 1개"가 문서 1개로 취급

CorpusDocs = List[List[str]]
```

### 2-2) 문서 내 위치 찾기 + 다음 문장

```python
def find_positions(doc: List[str], ctx: str) -> List[int]:
    return [i for i, s in enumerate(doc) if s == ctx]

def get_next_sentences_in_doc(doc: List[str], pos: int, window: int = 3) -> List[str]:
    out = []
    for j in range(1, window + 1):
        if pos + j < len(doc):
            out.append(doc[pos + j])
    return out
```

### 2-3) 마커 기반(접속어) 후속 구간 추출

```python
MARKERS = ["따라서", "그러므로", "즉", "결국", "하지만", "다만", "반면", "구체적으로", "말하자면", "예를 들어"]

def marker_followups(doc: List[str], pos: int, window: int = 3) -> List[str]:
    out = []
    for j in range(1, window + 1):
        if pos + j >= len(doc):
            break
        s = doc[pos + j].strip()
        # marker로 시작하거나 marker가 초반에 등장하면 후보로
        if any(s.startswith(m) or (m in s[:10]) for m in MARKERS):
            out.append(s)
    return out
```

### 2-4) 1-step 변형(최소형)

여기서는 “패러프레이즈 모델”까지 넣지 않고, 최소한의 변형만 합니다.

```python
def single_step_transforms(ctx: str) -> List[str]:
    # MVP: 접속어/정의 프레임을 붙여 의미 생성 방향을 강제
    prefixes = ["즉, ", "말하자면, ", "구체적으로, ", "따라서, ", "다만, "]
    return [p + ctx for p in prefixes]
```

### 2-5) 후보를 임베딩 유사도로 랭킹해서 top-k

여기서 **개념 코어 중심**을 써야 합니다(이게 “본질/응축”의 구현).

- `core_centroid`: 개념 C의 중심 벡터(앞에서 x0로 잡았던 것)
    

```python
def rank_by_core_similarity(
    candidates: List[str],
    embed: EmbeddingModel,
    core_centroid: np.ndarray,
) -> List[str]:
    c0 = l2_normalize(core_centroid)
    scored = []
    for s in candidates:
        v = embed.encode(s)
        sim = cosine(v, c0)
        scored.append((sim, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [s for _, s in scored]
```

### 2-6) 최종 propose_next_contexts (문서 단위 + 마커 + 변형 + 랭킹)

```python
def propose_next_contexts_v2(
    ctx: str,
    corpus_docs: CorpusDocs,
    embed: EmbeddingModel,
    core_centroid: np.ndarray,
    k_neighbors: int,
) -> List[str]:
    candidates: List[str] = []

    # 1) 담화 연쇄(같은 문서에서)
    for doc in corpus_docs:
        positions = find_positions(doc, ctx)
        for pos in positions:
            candidates += get_next_sentences_in_doc(doc, pos, window=3)
            candidates += marker_followups(doc, pos, window=3)

    # 2) 1-step 변형
    candidates += single_step_transforms(ctx)

    # 3) dedupe
    seen = set()
    deduped = []
    for s in candidates:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            deduped.append(s)

    # 4) rank by similarity to core
    ranked = rank_by_core_similarity(deduped, embed, core_centroid)
    return ranked[:k_neighbors]
```

---

## 3) 기존 `fit_lower_operator()`에 v2를 연결하는 최소 수정

핵심은 `core_centroid`(=x0) 계산을 먼저 하고, 그걸 v2에 넣는 겁니다.

```python
def fit_lower_operator_v2(
    concept: str,
    corpus_docs: CorpusDocs,
    embed: EmbeddingModel,
    cfg: SRConfig,
) -> LowerOperator:
    rng = np.random.default_rng(cfg.seed)

    # 1) concept contexts 수집 (문서 전체에서)
    flat = [s for doc in corpus_docs for s in doc]
    ctxs = sample_contexts_for_concept(concept, flat, n=cfg.k_contexts, rng=rng)

    # 2) core centroid (MVP: 전체 평균. 나중에 density core로 바꿔도 됨)
    X_ctx = np.stack([embed.encode(c) for c in ctxs], axis=0)
    core_centroid = np.mean(X_ctx, axis=0)

    # 3) transitions
    X_cols, Y_cols = [], []
    for ctx in ctxs:
        next_ctxs = propose_next_contexts_v2(
            ctx=ctx,
            corpus_docs=corpus_docs,
            embed=embed,
            core_centroid=core_centroid,
            k_neighbors=cfg.k_neighbors,
        )
        if not next_ctxs:
            continue

        x = embed.encode(ctx)
        y = np.mean(np.stack([embed.encode(nc) for nc in next_ctxs], axis=0), axis=0)

        X_cols.append(x)
        Y_cols.append(y)

    if not X_cols:
        raise ValueError("No transition pairs created. Check corpus_docs / markers / concept hits.")

    X = np.stack(X_cols, axis=1)
    Y = np.stack(Y_cols, axis=1)
    d, m = X.shape

    ones = np.ones((1, m))
    Xp = np.vstack([X, ones])

    B = ridge_closed_form(Xp, Y, lam=cfg.ridge_lambda)
    A = B[:, :d]
    b = B[:, d]

    sn = spectral_norm(A)
    target = 1.0 - cfg.contraction_eps
    if sn > target:
        A = (target / sn) * A

    return LowerOperator(A=A, b=b)
```

---

## 4) 이제 “실험”이 가능한 체크리스트(짧게)

- `corpus_docs`를 문서 단위로 준비
    
    - 예: 논문/메모/대화 1개 = doc 1개
        
- concept 노드 목록 `concept_nodes`는 처음엔 20~200개 정도가 적당
    
- 수렴 안 되면 거의 항상:
    
    1. 문서가 섞여서 담화 전이가 무의미하거나
        
    2. concept hit가 너무 적거나
        
    3. 노드 사전이 너무 작아서 (S(x))가 빈약하거나
        
    4. prune_theta가 너무 커서 W가 다 죽거나
        
    5. beta가 너무 작아서 a(x)가 평평해짐
        

---

원하시면 다음 단계로는 **“실제로 수렴을 잘 시키는 디폴트 파라미터”**를 잡아드리겠습니다.  
특히 (\eta,\beta,\theta,\lambda) 조합은 수렴/발산/쓸모없는 수렴(다 0으로 수렴)을 갈라서, 여기서 튜닝 규칙을 딱 정해두면 이후는 매우 빨라집니다.

# 디폴트 파라미터
네. “수렴이 잘 되는” 쪽으로 **디폴트 파라미터를 고정값 + 튜닝 규칙**까지 같이 드리겠습니다.  
(목표: _발산 방지_ + _W가 다 0이 되는 쓸모없는 수렴 방지_ + _과도한 고정(한 노드로 몰림) 방지_)

---

## 1) 바로 쓰는 디폴트(추천 세트)

### 코퍼스 규모별 기본값

- 문서/문장 수가 **적당히 있음(문장 3천~5만)** 기준
    

```python
cfg = SRConfig(
    k_contexts=2000,     # concept hit이 충분하면 2000
    k_neighbors=8,       # propose_next_contexts top-k
    ridge_lambda=1e-2,   # 안정적 기본
    contraction_eps=5e-3,# A를 확실히 수축시키기

    eta=0.10,            # W 업데이트 관성(너무 크면 요동)
    beta=12.0,           # a(x) 집중도 (softmax temperature)
    prune_theta=1e-4,    # W pruning (너무 크면 W가 다 죽음)

    t_max=60,
    eps_x=1e-4,
    eps_w=1e-4,
    seed=7
)
```

### 코퍼스가 작을 때(문장 < 3천)

```python
cfg = SRConfig(
    k_contexts=400,      # 과적합/노이즈 방지
    k_neighbors=6,
    ridge_lambda=5e-2,   # 더 강한 정규화
    contraction_eps=1e-2,# 더 강한 수축

    eta=0.15,            # 데이터 적을 때 W가 안 자라서 조금 올림
    beta=10.0,           # 너무 뾰족하면 과도하게 한 노드로 몰림
    prune_theta=5e-5,

    t_max=80,
    eps_x=1e-4,
    eps_w=1e-4,
    seed=7
)
```

---

## 2) “수렴이 잘 된다”의 3가지 실패 모드와 즉각 처방

### A. 발산/진동(수렴 안 함): dx, dW가 줄지 않음

**증상**

- `dx`가 줄지 않거나 들쭉날쭉
    
- `dW`도 계속 큼
    

**처방(우선순위대로)**

1. `contraction_eps` ↑ (더 수축)
    
    - `5e-3 → 1e-2`
        
2. `eta` ↓ (W 업데이트를 느리게)
    
    - `0.10 → 0.05`
        
3. `k_neighbors` ↓ (다음 컨텍스트 노이즈 줄임)
    
    - `8 → 4~6`
        
4. `ridge_lambda` ↑ (A 안정화)
    
    - `1e-2 → 5e-2`
        

---

### B. “쓸모없는 수렴”: W가 거의 전부 0으로 수렴

**증상**

- 수렴은 하는데 `W mass = sum(W)`가 거의 0
    
- 또는 pruning 이후 W가 대부분 0
    

**처방**

1. `prune_theta` ↓
    
    - `1e-4 → 1e-5`
        
2. `beta` ↑ (a(x) 분포를 더 뾰족하게)
    
    - `12 → 16`
        
3. `N(노드 수)` ↑ (노드 사전이 너무 작으면 활성 자체가 빈약함)
    
    - 최소 50~200 권장
        
4. `eta` ↑ (W가 자라도록)
    
    - `0.10 → 0.15` (단, 진동하면 다시 내리기)
        

---

### C. “한 노드로 몰리는 수렴”: W가 거의 랭크-1처럼 됨

**증상**

- `a(x)`가 거의 원-핫(한 개만 0.99)
    
- W가 특정 노드 중심으로만 강해짐
    

**처방**

1. `beta` ↓
    
    - `12 → 8~10`
        
2. `prune_theta` ↑ 약간 (너무 약한 엣지 정리)
    
    - `1e-4 → 3e-4` (단, B로 넘어가면 안 됨)
        
3. 노드 프로토타입 (u_j) 품질 개선
    
    - 각 노드의 컨텍스트 샘플을 늘리거나, 다의 노드는 분리
        

---

## 3) 디폴트로 “자동 튜닝” 규칙(코드로 넣기 좋은 형태)

실험 한 번 돌리고 아래 규칙을 적용하면 됩니다.

- `W mass < 0.05` 이면
    
    - `prune_theta *= 0.1`
        
    - `beta += 2`
        
- `max(a(x*)) > 0.98` 이면
    
    - `beta -= 2`
        
- `dx 마지막 5스텝 평균 > dx 첫 5스텝 평균 * 0.7` 이면(안 줄어듦)
    
    - `contraction_eps *= 2`
        
    - `eta *= 0.7`
        
    - `ridge_lambda *= 5`
        

---

## 4) “좋은 디폴트”의 감각적 목표치(로그에서 확인)

수렴이 “의미 있게” 잘 되면 보통:

- `dx`: 초반 크게 떨어지고, 후반 `1e-4 ~ 1e-5` 근처
    
- `dW`: 초반 떨어지고, 후반 `1e-4` 근처
    
- `W mass`: 0이 아니라 **0.2 ~ 3.0** 정도(스케일 따라 다름)
    
- `a(x*)`: 상위 3~10개가 의미 있게 분포(한 개 독점 아님)
    

---

## 5) 결론: 지금 당장 시작 세트

가장 무난한 한 세트만 찍으면 이겁니다.

- `eta=0.10, beta=12.0, prune_theta=1e-4, ridge_lambda=1e-2, contraction_eps=5e-3, k_neighbors=8, k_contexts=2000`
    

이걸로 돌리고, 위 “실패 모드”에 따라 조정하면 됩니다.

원하시면, 사용하실 **노드 사전(concept_nodes) 구성 규칙**(어떤 노드를 넣어야 W가 의미 있게 자라는지)도 바로 고정해드리겠습니다. 노드 사전 설계가 사실상 (\Phi)의 입력 공간을 결정해서, 수렴 품질에 크게 영향 줍니다.

# 노드 사전(concept_nodes) 구성 규칙
좋습니다. 노드 사전은 **(\Phi)의 상태공간 (R) 자체**라서, 여기서 규칙을 고정하면 이후 실험이 훨씬 안정됩니다. 아래는 **MVP에서 바로 쓰는 “고정 규칙”**입니다. (길게 늘어놓지 않고, 실무적으로 딱 돌아가게)

---

## 0) 목표 정의

노드 사전 ( \mathcal{N}={n_1,\dots,n_N})는

- 너무 작으면 (S(x))가 빈약해서 **W가 0으로 수렴**
    
- 너무 크면 프로토타입이 부실해져 **노이즈/몰림/진동**
    

그래서 “의미 있는 수렴”을 위한 목표는 이겁니다.

> **(i) 중심부를 커버하는 노드 + (ii) 경계/대립을 커버하는 노드 + (iii) 메타-관계 노드**  
> 이 3층이 함께 있어야 (W)가 구조를 갖고 응축합니다.

---

## 1) 노드 사전 크기 고정(디폴트)

- 코퍼스가 충분(문장 3천 이상): **N = 120** (권장 범위 80~200)
    
- 코퍼스 작음(문장 3천 미만): **N = 60** (권장 범위 40~100)
    

이걸 **기본값**으로 고정합니다.

---

## 2) 구성 비율 고정(핵심 규칙)

노드 사전은 아래 5개 “버킷”으로 만들고, 비율을 고정합니다.

## Bucket A. 타깃 개념의 코어 근접 노드 (40%)

- 타깃 개념 (C)와 **가장 자주 같이 등장**하거나,
    
- 임베딩에서 **가장 가까운** 개념들
    

> 목적: 중심부를 잡아서 (a(x))가 안정적으로 분포하도록 함

**규칙**

- “C가 포함된 문장”에서 공기어/명사구를 추출
    
- 빈도 상위 + 임베딩 유사도 상위를 교집합으로 택함
    

---

## Bucket B. 상위/하위(범주) 노드 (15%)

- 상위 개념(hypernym) 8~12개
    
- 하위 개념(hyponym) 8~12개
    

> 목적: 의미공간의 “축”을 만들어서 발산/몰림을 줄임

**규칙(최소형)**

- 패턴 기반으로 뽑습니다.
    
    - “C는 ~이다”
        
    - “C의 한 종류는”
        
    - “C에는 ~가 있다/포함된다”
        

---

## Bucket C. 대립/경계 노드 (15%)

- 반의어, 충돌, 대비, 예외를 만드는 개념들
    

> 목적: 의미개방성(꼬리)을 “경계”로 수렴시키는 장치

**규칙**

- “하지만/다만/반면/그러나” 근처에서 C와 함께 나오는 핵심 명사/형용사
    
- “C가 아니다/반대로” 패턴에서 나오는 개념
    

---

## Bucket D. 관계 타입(메타) 노드 (20%)

여기서가 사용자님의 강점(메타윤리 연산자)과 맞물립니다.

> 규범성에서 잡아낸 ‘관계 개념’을 노드로 직접 넣습니다.

예시(최소 세트)

- 정당화 / 근거 / 이유
    
- 의무 / 허용 / 금지
    
- 조건 / 예외
    
- 목적 / 수단
    
- 원인 / 결과
    
- 함의 / 모순 / 일관성
    
- 정의 / 구체화 / 사례
    

> 목적: (\Phi)가 단순 동시활성(aᵢaⱼ)만으로도 “구조”를 갖게 함

**규칙**

- 버킷 D는 **사용자 고정 리스트**로 시작하는 게 가장 좋습니다.
    
- 코퍼스에서 발견되면 강화, 안 나오면 약화되며 자연스럽게 걸러집니다.
    

---

## Bucket E. “앵커(Anchor)” 노드 (10%)

- 아주 일반적이고 안정적인 기준점 개념들
    

예:

- 사실 / 평가
    
- 설명 / 예측
    
- 언어 / 수학 / 모델
    
- 사람 / 행위
    
- 시간 / 공간
    
- 시스템 / 구조
    

> 목적: 코퍼스가 바뀌어도 좌표계가 무너지지 않게 하는 기준점

---

## 3) “선정 규칙”을 완전 고정 (점수 함수)

후보 노드 (w)에 대해 다음 점수를 계산하고, 버킷별로 상위권을 뽑습니다.

## A(코어 근접) 점수

[  
\text{score}_A(w)=\alpha,\text{freq}(w\mid C) + (1-\alpha),\cos(u_C, u_w)  
]

- (\alpha = 0.7) 고정
    
- (\text{freq}(w\mid C)): C 포함 문장 내 빈도(정규화)
    

## C(대립/경계) 점수

[  
\text{score}_C(w)=\text{freq}(w\mid \text{contrast}(C))  
]

- contrast(C): “하지만/다만/반면/그러나” 구간에서 C와 함께 등장
    

B는 패턴으로 뽑아 **점수 없이** 상위/하위 각 10개로 고정해도 됩니다.

D, E는 **고정 리스트**를 먼저 넣고, 코퍼스에서 등장하면 유지(프로토타입 생성 가능), 전혀 안 나오면 다음 라운드에서 제거.

---

## 4) 품질 기준(제거 규칙) 고정

노드 사전이 망가지는 가장 흔한 이유는 “프로토타입이 빈약한 노드”입니다.

노드 (n)에 대해:

- 해당 노드 컨텍스트 샘플 수가 (< 30) 이면 제거  
    (코퍼스 작은 경우 (< 10) )
    
- 또는 프로토타입 벡터 분산이 너무 크면(다의성이 심함) 제거/분해
    

MVP에서는 샘플 수 기준만으로 충분합니다.

---

## 5) 완성된 “고정 레시피”(한 문단)

1. 타깃 개념 C 문장들에서 후보 추출
    
2. Bucket A: 빈도+유사도 혼합 점수 상위 40%
    
3. Bucket B: 정의/종속 패턴으로 상위·하위 각 10개
    
4. Bucket C: 대립 마커 구간 빈도 상위 15%
    
5. Bucket D: 관계-메타 고정 리스트 20%
    
6. Bucket E: 앵커 고정 리스트 10%
    
7. 각 노드별 컨텍스트 샘플이 일정 수 미만이면 제거하고, 빈칸은 A에서 보충
    
8. 최종 N=120(또는 60)로 확정
    

이게 “노드 사전 구성 규칙”의 고정 버전입니다.

---

## 바로 쓸 수 있는 고정 리스트(버킷 D/E 최소 세트)

### D(관계/규범성 메타 노드) — 24개

- 정당화, 근거, 이유, 설명
    
- 의무, 허용, 금지, 권한, 책임
    
- 조건, 예외, 원칙, 기준
    
- 목적, 수단, 효과, 결과, 원인
    
- 함의, 모순, 일관성, 충돌
    
- 정의, 구체화, 사례, 해석
    

### E(앵커) — 12개

- 사실, 평가, 규칙, 판단
    
- 언어, 의미, 모델, 수식
    
- 시간, 구조, 시스템, 관계
    

(코퍼스에 안 나오는 단어는 자연히 프로토타입이 약해져서 다음 라운드에서 떨어집니다.)

---

원하시면 다음 단계로, 위 규칙을 그대로 옮긴 **`build_concept_nodes()` 함수(파이썬 코드)**를 만들어 드리겠습니다.  
입력: (concept C, corpus_docs, embed) / 출력: concept_nodes 리스트 + 각 버킷 구성 로그.

# 노드 구성 규칙 모듈 파이썬 코드
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
import re
import math
import numpy as np


# -------------------------
# Minimal text utilities
# -------------------------

_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9_]+")  # Korean blocks + basic alnum

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)

def dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, eps)

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (max(na, eps) * max(nb, eps)))


# -------------------------
# Config + Output
# -------------------------

@dataclass
class NodeDictConfig:
    # target dictionary size
    N: int = 120

    # bucket ratios
    ratio_A_core: float = 0.40
    ratio_B_taxonomy: float = 0.15
    ratio_C_contrast: float = 0.15
    ratio_D_meta: float = 0.20
    ratio_E_anchor: float = 0.10

    # scoring mix for A bucket
    alpha_freq: float = 0.70  # score_A = alpha*freq + (1-alpha)*cos

    # sample sizes / thresholds
    k_contexts_for_C: int = 2000          # sample contexts containing C for candidate mining
    proto_min_contexts: int = 30          # min contexts per node to keep (large corpus)
    proto_min_contexts_small: int = 10    # if corpus is small, use this
    small_corpus_sentence_threshold: int = 3000

    # extraction limits
    max_candidates_A: int = 5000
    max_candidates_C: int = 3000

    # markers
    contrast_markers: Tuple[str, ...] = ("하지만", "다만", "반면", "그러나", "그런데")
    refine_markers: Tuple[str, ...] = ("즉", "말하자면", "구체적으로", "예를 들어", "다시 말해")

    # taxonomy patterns (very rough MVP)
    # You can extend/replace these with proper pattern matching / parsing.
    taxonomy_markers_upper: Tuple[str, ...] = ("는", "은", "란", "이란")  # for "C는 X이다"
    taxonomy_is_predicates: Tuple[str, ...] = ("이다", "다", "입니다", "였다", "라고", "이라고")
    taxonomy_markers_lower: Tuple[str, ...] = ("종류", "예", "포함", "중", "가령")

    # fixed lists
    meta_nodes: Tuple[str, ...] = (
        "정당화", "근거", "이유", "설명",
        "의무", "허용", "금지", "권한", "책임",
        "조건", "예외", "원칙", "기준",
        "목적", "수단", "효과", "결과", "원인",
        "함의", "모순", "일관성", "충돌",
        "정의", "구체화", "사례", "해석",
    )
    anchor_nodes: Tuple[str, ...] = (
        "사실", "평가", "규칙", "판단",
        "언어", "의미", "모델", "수식",
        "시간", "구조", "시스템", "관계",
    )


@dataclass
class NodeDictBuildLog:
    counts: Dict[str, int]
    removed_for_low_contexts: List[str]
    bucket_nodes: Dict[str, List[str]]
    notes: List[str]


# -------------------------
# Corpus helpers
# -------------------------

def flatten_docs(corpus_docs: Sequence[Sequence[str]]) -> List[str]:
    return [s for doc in corpus_docs for s in doc]

def sample_contexts_for_concept(
    concept: str,
    sentences: Sequence[str],
    k: int,
    rng: np.random.Generator,
) -> List[str]:
    hits = [s for s in sentences if concept in s]
    if not hits:
        return []
    if len(hits) <= k:
        return hits
    idx = rng.choice(len(hits), size=k, replace=False)
    return [hits[i] for i in idx]

def extract_window_around_markers(
    concept: str,
    sentences: Sequence[str],
    markers: Tuple[str, ...],
    window: int = 1,
) -> List[str]:
    """
    Extract sentences that contain concept and (marker in same sentence) OR
    take neighbor sentences around a sentence containing both.
    Minimal: same sentence only, plus optional adjacent.
    """
    out = []
    n = len(sentences)
    for i, s in enumerate(sentences):
        if concept not in s:
            continue
        if any(m in s for m in markers):
            out.append(s)
            for j in range(1, window + 1):
                if i + j < n:
                    out.append(sentences[i + j])
                if i - j >= 0:
                    out.append(sentences[i - j])
    return out


# -------------------------
# Embedding interface expected
# -------------------------

class EmbeddingModel:
    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError


# -------------------------
# Prototype availability check
# -------------------------

def count_node_contexts(node: str, sentences: Sequence[str]) -> int:
    # MVP: count sentences containing the node substring
    return sum(1 for s in sentences if node in s)

def build_prototype_vector(
    node: str,
    sentences: Sequence[str],
    embed: EmbeddingModel,
    max_samples: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    rng = rng or np.random.default_rng(7)
    ctxs = [s for s in sentences if node in s]
    if not ctxs:
        return None
    if len(ctxs) > max_samples:
        idx = rng.choice(len(ctxs), size=max_samples, replace=False)
        ctxs = [ctxs[i] for i in idx]
    vecs = [embed.encode(s) for s in ctxs]
    u = np.mean(np.stack(vecs, axis=0), axis=0)
    return l2_normalize(u)


# -------------------------
# Taxonomy heuristics (MVP)
# -------------------------

def extract_taxonomy_candidates_upper(concept: str, contexts: Sequence[str], limit: int = 30) -> List[str]:
    """
    Try to get 'upper' candidates roughly from patterns like:
      "C는 X이다" -> X
    Very rough: look for "... concept ... 는/은 ... {token} ... 이다"
    """
    cands = []
    for s in contexts:
        if concept not in s:
            continue
        # simple split at concept
        after = s.split(concept, 1)[1]
        # look for "는/은/란/이란" then take next token chunk
        m = re.search(r"(는|은|란|이란)\s*([가-힣A-Za-z0-9_]{2,})", after)
        if m:
            cand = m.group(2)
            if cand != concept:
                cands.append(cand)
        if len(cands) >= limit:
            break
    return dedupe_keep_order(cands)

def extract_taxonomy_candidates_lower(concept: str, contexts: Sequence[str], limit: int = 30) -> List[str]:
    """
    Try to get 'lower' candidates from patterns like:
      "C에는 A, B가 포함된다", "C의 종류로 A"
    MVP: pick tokens near words like '종류/예/포함/가령' in sentences with concept.
    """
    cands = []
    for s in contexts:
        if concept not in s:
            continue
        if any(k in s for k in ("종류", "예", "포함", "가령", "예컨대")):
            toks = tokenize(s)
            # add tokens excluding very short ones and the concept
            for t in toks:
                if t == concept:
                    continue
                if len(t) >= 2:
                    cands.append(t)
        if len(cands) >= limit:
            break
    return dedupe_keep_order(cands)


# -------------------------
# Main builder
# -------------------------

def build_concept_nodes(
    concept: str,
    corpus_docs: Sequence[Sequence[str]],
    embed: EmbeddingModel,
    cfg: Optional[NodeDictConfig] = None,
    seed: int = 7,
) -> Tuple[List[str], NodeDictBuildLog]:
    """
    Build concept_nodes list following fixed bucket rules:

    Buckets:
      A (40%): core-near nodes by score_A = alpha*freq(C-context) + (1-alpha)*cos(uC, u_w)
      B (15%): taxonomy nodes (upper/hyper + lower/hypo) from pattern heuristics
      C (15%): contrast/boundary nodes from contrast-marker contexts frequency
      D (20%): fixed meta nodes (relation/normativity)
      E (10%): fixed anchors

    Then filter nodes with insufficient contexts for prototype building.
    Finally, fill remaining slots from A bucket (next best).
    """
    cfg = cfg or NodeDictConfig()
    rng = np.random.default_rng(seed)

    sentences = flatten_docs(corpus_docs)
    total_sent = len(sentences)

    # Choose min contexts threshold by corpus size
    proto_min = cfg.proto_min_contexts_small if total_sent < cfg.small_corpus_sentence_threshold else cfg.proto_min_contexts

    notes: List[str] = []
    if total_sent < cfg.small_corpus_sentence_threshold:
        notes.append(f"Small corpus detected (sentences={total_sent}); using proto_min_contexts={proto_min}.")
    else:
        notes.append(f"Corpus size OK (sentences={total_sent}); using proto_min_contexts={proto_min}.")

    # --- collect contexts for concept C ---
    C_ctxs = sample_contexts_for_concept(concept, sentences, k=cfg.k_contexts_for_C, rng=rng)
    if not C_ctxs:
        raise ValueError(f"No contexts found containing concept='{concept}'. Provide corpus where it appears.")

    # --- build u_C ---
    uC = build_prototype_vector(concept, sentences, embed, max_samples=300, rng=rng)
    if uC is None:
        raise ValueError("Failed to build prototype for concept C (unexpected).")

    # --- candidate mining for A ---
    # freq(w|C): count occurrences in C-contexts (token-based)
    freq: Dict[str, int] = {}
    for s in C_ctxs:
        for t in tokenize(s):
            if t == concept:
                continue
            if len(t) < 2:
                continue
            freq[t] = freq.get(t, 0) + 1

    # Keep top candidates by raw frequency first (cap)
    freq_items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    freq_items = freq_items[: cfg.max_candidates_A]
    candidates_A = [w for w, _ in freq_items]

    # build prototypes for top-A candidates (for cosine)
    # for speed, only compute for a prefix; still enough for MVP
    max_proto = min(len(candidates_A), 600)  # cap to control runtime
    proto_cache: Dict[str, np.ndarray] = {}
    for w in candidates_A[:max_proto]:
        u = build_prototype_vector(w, sentences, embed, max_samples=120, rng=rng)
        if u is not None:
            proto_cache[w] = u

    # compute score_A
    # normalize freq by max frequency
    maxf = max(freq.values()) if freq else 1
    scored_A = []
    for w in candidates_A:
        fw = freq.get(w, 0) / maxf
        if w in proto_cache:
            sim = max(0.0, cosine(uC, proto_cache[w]))  # clamp negative to 0 for stability
        else:
            # if no proto, treat similarity as 0
            sim = 0.0
        scoreA = cfg.alpha_freq * fw + (1.0 - cfg.alpha_freq) * sim
        scored_A.append((scoreA, w))
    scored_A.sort(reverse=True, key=lambda x: x[0])

    # --- bucket C candidates (contrast contexts frequency) ---
    contrast_ctxs = extract_window_around_markers(concept, C_ctxs, cfg.contrast_markers, window=0)
    freqC: Dict[str, int] = {}
    for s in contrast_ctxs:
        for t in tokenize(s):
            if t == concept:
                continue
            if len(t) < 2:
                continue
            freqC[t] = freqC.get(t, 0) + 1
    candC_items = sorted(freqC.items(), key=lambda kv: kv[1], reverse=True)[: cfg.max_candidates_C]
    candidates_C = [w for w, _ in candC_items]

    # --- bucket B (taxonomy) ---
    # use both C-contexts and refined-marker contexts to increase chance
    refine_ctxs = extract_window_around_markers(concept, C_ctxs, cfg.refine_markers, window=0)
    tax_base = C_ctxs + refine_ctxs
    B_upper = extract_taxonomy_candidates_upper(concept, tax_base, limit=40)
    B_lower = extract_taxonomy_candidates_lower(concept, tax_base, limit=80)

    # Clean taxonomy candidates (remove too generic / concept itself)
    def _clean_tax(xs: List[str]) -> List[str]:
        out = []
        for x in xs:
            if x == concept:
                continue
            if len(x) < 2:
                continue
            out.append(x)
        return dedupe_keep_order(out)

    B_upper = _clean_tax(B_upper)
    B_lower = _clean_tax(B_lower)

    # --- bucket D/E fixed lists ---
    bucket_D = list(cfg.meta_nodes)
    bucket_E = list(cfg.anchor_nodes)

    # --- compute target counts per bucket ---
    N = cfg.N
    nA = int(round(N * cfg.ratio_A_core))
    nB = int(round(N * cfg.ratio_B_taxonomy))
    nC = int(round(N * cfg.ratio_C_contrast))
    nD = int(round(N * cfg.ratio_D_meta))
    # remainder to E to ensure exact N
    nE = N - (nA + nB + nC + nD)
    if nE < 0:
        # if rounding overshoots, trim A first
        nE = max(0, nE)
        # recompute conservative
        nA = max(0, N - (nB + nC + nD + nE))

    notes.append(f"Bucket sizes: A={nA}, B={nB}, C={nC}, D={nD}, E={nE} (N={N}).")

    # --- pick from each bucket (pre-filter) ---
    # A: take top scored
    bucket_A = [w for _, w in scored_A[: max(nA * 3, nA)]]  # take extra for later fill/filter
    # B: split upper/lower half-half
    nB_upper = nB // 2
    nB_lower = nB - nB_upper
    bucket_B = (B_upper[:nB_upper] + B_lower[:nB_lower])
    # C: take top by contrast frequency
    bucket_C = candidates_C[: max(nC * 3, nC)]
    # D/E: take fixed
    bucket_D = bucket_D[:nD]
    bucket_E = bucket_E[:nE]

    # Always include the target concept itself as a node (anchor for interpretation)
    # Count it in A bucket effectively (doesn't matter, we enforce final N)
    pre_nodes = [concept] + bucket_A[:nA] + bucket_B + bucket_C[:nC] + bucket_D + bucket_E
    pre_nodes = dedupe_keep_order(pre_nodes)

    # --- filter by prototype availability (min contexts) ---
    removed: List[str] = []
    kept: List[str] = []
    for node in pre_nodes:
        cnt = count_node_contexts(node, sentences)
        if cnt < proto_min:
            # Keep the target concept regardless, unless it really doesn't exist (already checked)
            if node == concept:
                kept.append(node)
                continue
            removed.append(node)
            continue
        kept.append(node)

    # After filtering, we may be below N: fill from A (next best) and then C, then B
    kept_set = set(kept)

    def _fill_from(cands: List[str], limit: int):
        nonlocal kept, kept_set, removed
        for w in cands:
            if len(kept) >= limit:
                break
            if w in kept_set:
                continue
            if count_node_contexts(w, sentences) < proto_min:
                removed.append(w)
                continue
            kept.append(w)
            kept_set.add(w)

    # Ensure we fill up to N
    # Prioritize: A remainder -> C remainder -> B remainder -> D/E remainder (if any left)
    A_remainder = [w for _, w in scored_A if w not in kept_set]
    C_remainder = [w for w in candidates_C if w not in kept_set]
    B_remainder = [w for w in (B_upper + B_lower) if w not in kept_set]
    D_remainder = [w for w in cfg.meta_nodes if w not in kept_set]
    E_remainder = [w for w in cfg.anchor_nodes if w not in kept_set]

    _fill_from(A_remainder, N)
    _fill_from(C_remainder, N)
    _fill_from(B_remainder, N)
    _fill_from(D_remainder, N)
    _fill_from(E_remainder, N)

    # If still below N, relax threshold (only if absolutely necessary)
    if len(kept) < N:
        notes.append(
            f"Warning: only {len(kept)} nodes met proto_min_contexts={proto_min}. "
            f"Relaxing threshold to fill remaining."
        )
        relaxed_min = max(3, proto_min // 3)
        for w in A_remainder + C_remainder + B_remainder:
            if len(kept) >= N:
                break
            if w in kept_set:
                continue
            if count_node_contexts(w, sentences) < relaxed_min:
                continue
            kept.append(w)
            kept_set.add(w)

    # Trim to N
    concept_nodes = kept[:N]

    # Bucket reporting (what actually survived)
    bucket_nodes = {
        "A_core": [x for x in concept_nodes if x in set(bucket_A)],
        "B_taxonomy": [x for x in concept_nodes if x in set(bucket_B)],
        "C_contrast": [x for x in concept_nodes if x in set(bucket_C)],
        "D_meta": [x for x in concept_nodes if x in set(cfg.meta_nodes)],
        "E_anchor": [x for x in concept_nodes if x in set(cfg.anchor_nodes)],
        "Concept": [concept] if concept in concept_nodes else [],
    }

    log = NodeDictBuildLog(
        counts={
            "sentences_total": total_sent,
            "contexts_for_C": len(C_ctxs),
            "proto_min_contexts_used": proto_min,
            "pre_nodes": len(pre_nodes),
            "removed_for_low_contexts": len(removed),
            "final_nodes": len(concept_nodes),
        },
        removed_for_low_contexts=dedupe_keep_order(removed),
        bucket_nodes=bucket_nodes,
        notes=notes,
    )
    return concept_nodes, log


# -------------------------
# Example usage (requires real EmbeddingModel)
# -------------------------
if __name__ == "__main__":
    # Dummy embedding for demonstration only
    class DummyEmbed(EmbeddingModel):
        def __init__(self, d: int = 32):
            self.d = d

        def encode(self, text: str) -> np.ndarray:
            h = abs(hash(text)) % (10**9)
            rng = np.random.default_rng(h)
            return rng.normal(size=(self.d,)).astype(float)

    corpus_docs = [
        [
            "규범성은 개념과 개념 사이의 관계를 정의하는 개념이다.",
            "따라서 자기지시 연산자는 규범성에서 먼저 잡힌다.",
            "하지만 자연언어 형식화는 의미개방성 때문에 어렵다.",
            "즉 의미공간에서 분포가 응축되는 지점을 잡아야 한다.",
        ],
        [
            "본질이란 의미개방성이 가장 강하게 응축된 구간이다.",
            "반면 주변부는 꼬리로 퍼져 무시 가능해진다.",
            "구체적으로는 의미 벡터 분포의 밀도 중심을 찾는다.",
        ],
    ]

    embed = DummyEmbed()
    cfg = NodeDictConfig(N=60)  # small for demo
    nodes, log = build_concept_nodes("본질", corpus_docs, embed, cfg=cfg)

    print("N nodes:", len(nodes))
    print("First 30 nodes:", nodes[:30])
    print("Counts:", log.counts)
    print("Notes:", log.notes)
    print("Bucket sizes (actual):", {k: len(v) for k, v in log.bucket_nodes.items()})
    print("Removed (first 20):", log.removed_for_low_contexts[:20])
```

### 사용 방법(딱 핵심만)

- 지금 파일에서 `DummyEmbed`를 **실제 임베딩 모델**로 바꾸시고,
    
- `corpus_docs`를 “문서별 문장 리스트”로 넣은 다음,
    
- `build_concept_nodes(concept, corpus_docs, embed)` 호출하시면 됩니다.
    

원하시면 다음 단계로, 이 `concept_nodes`를 그대로 받아서 **(1) 프로토타입 (u_j) 계산 + (2) formalize_concept_mvp 실행**까지 한 번에 묶는 `run_pipeline()` 함수도 만들어드리겠습니다.

# run_pipeline() 함수

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict
import numpy as np

# Assumes the following are already defined in your codebase (from previous snippets):
# - EmbeddingModel
# - SRConfig
# - NodeDictConfig, NodeDictBuildLog
# - build_concept_nodes
# - build_prototypes
# - fit_lower_operator_v2  (the improved version using propose_next_contexts_v2)
# - UpperOperator, LowerOperator
# - rollout, RolloutResult
# - flatten_docs, sample_contexts_for_concept
# - spectral_norm

# If you kept older names, adjust imports accordingly.


@dataclass
class PipelineArtifacts:
    concept: str
    concept_nodes: List[str]
    node_log: NodeDictBuildLog

    # prototypes
    U: np.ndarray                      # (N, d)
    node_index: Dict[str, int]

    # operators
    lower: "LowerOperator"             # (A,b)
    upper: "UpperOperator"             # (eta,beta,theta,U)

    # initial states
    x0: np.ndarray                     # (d,)
    W0: np.ndarray                     # (N,N)

    # rollout
    result: "RolloutResult"


def run_pipeline(
    concept: str,
    corpus_docs: Sequence[Sequence[str]],
    embed: "EmbeddingModel",
    sr_cfg: Optional["SRConfig"] = None,
    node_cfg: Optional["NodeDictConfig"] = None,
    seed: int = 7,
) -> PipelineArtifacts:
    """
    End-to-end MVP pipeline:

    1) Build node dictionary concept_nodes following fixed bucket rules.
    2) Build prototypes U for these nodes.
    3) Fit lower operator f: x_{t+1} = A x_t + b (v2, doc-aware next-context proposals).
    4) Build upper operator Φ with U.
    5) Initialize x0 as centroid of concept contexts; W0 as zeros.
    6) Rollout until convergence.

    Returns all artifacts + logs.
    """
    sr_cfg = sr_cfg or SRConfig(seed=seed)
    node_cfg = node_cfg or NodeDictConfig()

    rng = np.random.default_rng(seed)

    # 1) Node dictionary
    concept_nodes, node_log = build_concept_nodes(
        concept=concept,
        corpus_docs=corpus_docs,
        embed=embed,
        cfg=node_cfg,
        seed=seed,
    )

    # 2) Prototypes U
    flat_sentences = flatten_docs(corpus_docs)
    U, node_index = build_prototypes(
        concept_nodes=concept_nodes,
        corpus=flat_sentences,   # build_prototypes expects flat corpus in earlier snippet
        embed=embed,
        cfg=sr_cfg,
    )

    # 3) Fit lower operator (doc-aware)
    lower = fit_lower_operator_v2(
        concept=concept,
        corpus_docs=corpus_docs,
        embed=embed,
        cfg=sr_cfg,
    )

    # 4) Upper operator
    upper = UpperOperator(
        U=U,
        eta=sr_cfg.eta,
        beta=sr_cfg.beta,
        prune_theta=sr_cfg.prune_theta,
    )

    # 5) Initial state x0 (centroid of concept contexts)
    ctxs = sample_contexts_for_concept(
        concept=concept,
        sentences=flat_sentences,
        n=min(sr_cfg.k_contexts, 500),
        rng=rng,
    )
    if not ctxs:
        raise ValueError(f"No contexts found for concept '{concept}' while initializing x0.")

    X0 = np.mean(np.stack([embed.encode(c) for c in ctxs], axis=0), axis=0)
    x0 = X0

    # Initial W0
    N = len(concept_nodes)
    W0 = np.zeros((N, N), dtype=float)

    # 6) Rollout
    result = rollout(
        f=lower,
        Phi=upper,
        x0=x0,
        W0=W0,
        cfg=sr_cfg,
    )

    return PipelineArtifacts(
        concept=concept,
        concept_nodes=concept_nodes,
        node_log=node_log,
        U=U,
        node_index=node_index,
        lower=lower,
        upper=upper,
        x0=x0,
        W0=W0,
        result=result,
    )


# -------------------------
# Optional: Convenience report
# -------------------------

def summarize_pipeline(art: PipelineArtifacts, top_k_edges: int = 30) -> str:
    """
    Create a compact textual summary (no plotting).
    """
    lines = []
    lines.append(f"Concept: {art.concept}")
    lines.append(f"Nodes: {len(art.concept_nodes)}")
    lines.append(f"Converged: {art.result.converged} (steps={art.result.steps})")
    lines.append(f"A spectral norm: {spectral_norm(art.lower.A):.6f}")
    if art.result.history["dx"]:
        lines.append(f"dx last: {art.result.history['dx'][-1]:.6g}")
    if art.result.history["dW"]:
        lines.append(f"dW last: {art.result.history['dW'][-1]:.6g}")
    lines.append(f"W mass: {float(np.sum(art.result.W_star)):.6g}")
    lines.append("")
    lines.append("Bucket sizes (actual):")
    for k, v in art.node_log.bucket_nodes.items():
        lines.append(f"  - {k}: {len(v)}")
    lines.append("")

    # Top edges by weight
    W = art.result.W_star
    if W.size:
        idx = np.dstack(np.unravel_index(np.argsort(W.ravel())[::-1], W.shape))[0]
        edges = []
        for i, j in idx:
            w = W[i, j]
            if w <= 0:
                break
            ni = art.concept_nodes[int(i)]
            nj = art.concept_nodes[int(j)]
            if ni == nj:
                continue
            edges.append((w, ni, nj))
            if len(edges) >= top_k_edges:
                break

        lines.append(f"Top {min(top_k_edges, len(edges))} edges (w, from -> to):")
        for w, ni, nj in edges:
            lines.append(f"  {w:.6g}: {ni} -> {nj}")

    return "\n".join(lines)


# -------------------------
# Example (requires real EmbeddingModel)
# -------------------------
if __name__ == "__main__":
    # Dummy embed just for demonstration
    class DummyEmbed(EmbeddingModel):
        def __init__(self, d: int = 32):
            self.d = d
        def encode(self, text: str) -> np.ndarray:
            h = abs(hash(text)) % (10**9)
            rng = np.random.default_rng(h)
            return rng.normal(size=(self.d,)).astype(float)

    corpus_docs = [
        [
            "본질이란 의미개방성이 가장 강하게 응축된 구간이다.",
            "즉 의미공간에서 분포가 밀집되는 중심부를 말한다.",
            "하지만 주변부는 꼬리로 퍼져 중요도가 낮아진다.",
        ],
        [
            "자기지시 모델링의 병목은 자연언어 개념의 형식화다.",
            "따라서 의미 생성 규칙을 연산으로 잡아야 한다.",
            "반면 규범성은 개념 간 관계를 정의하므로 비교적 쉽다.",
        ]
    ]

    embed = DummyEmbed()

    # Choose defaults (adjust as needed)
    sr_cfg = SRConfig(
        k_contexts=400,
        k_neighbors=6,
        ridge_lambda=5e-2,
        contraction_eps=1e-2,
        eta=0.15,
        beta=10.0,
        prune_theta=5e-5,
        t_max=80,
        eps_x=1e-4,
        eps_w=1e-4,
        seed=7
    )
    node_cfg = NodeDictConfig(N=60)

    art = run_pipeline(
        concept="본질",
        corpus_docs=corpus_docs,
        embed=embed,
        sr_cfg=sr_cfg,
        node_cfg=node_cfg,
        seed=7,
    )

    print(summarize_pipeline(art, top_k_edges=15))
```

원하시면, 다음 단계로 **“수렴이 의미 있게 되었는지” 자동 판정**(예: `W mass` 범위, `max(a(x*))` 과몰림 여부, 상위 엣지의 다양성 지표 등)을 함수로 추가해드릴 수 있습니다.

# 수렴 자동 판정 함수
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# Assumes these exist from your pipeline code:
# - PipelineArtifacts
# - l2_normalize, cosine (or you can use simple dot / norms)
# - spectral_norm


@dataclass
class QualityReport:
    ok: bool
    verdict: str
    metrics: Dict[str, float]
    flags: List[str]
    suggestions: List[str]


def _topk_nonzero(a: np.ndarray, k: int) -> np.ndarray:
    if a.size == 0:
        return a
    idx = np.argsort(a)[::-1]
    top = a[idx[:k]]
    return top

def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    p = p / max(float(np.sum(p)), eps)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def _effective_support(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    exp(entropy) approximates effective number of active components.
    """
    return float(np.exp(_entropy(p, eps=eps)))

def _calc_activation_distribution(art: "PipelineArtifacts") -> np.ndarray:
    """
    Compute a(x*) = softmax(beta * cos(x*, u_j))
    using the UpperOperator already present.
    """
    x_star = art.result.x_star
    a = art.upper.a(x_star)  # relies on UpperOperator.a()
    return np.asarray(a, dtype=float)

def _edge_stats(W: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns: mass, max_edge, density (fraction of nonzero entries)
    """
    mass = float(np.sum(W))
    max_edge = float(np.max(W)) if W.size else 0.0
    density = float(np.mean(W > 0)) if W.size else 0.0
    return mass, max_edge, density

def _top_edge_diversity(art: "PipelineArtifacts", top_k: int = 50) -> float:
    """
    Diversity proxy for top edges:
      (# unique nodes appearing in top-k edges) / min(N, 2*top_k)

    Higher is better (not all edges attached to one hub).
    """
    W = art.result.W_star
    if W.size == 0:
        return 0.0
    flat = W.ravel()
    idx = np.argsort(flat)[::-1]
    used = set()
    cnt = 0
    for id_ in idx:
        w = flat[id_]
        if w <= 0:
            break
        i = int(id_ // W.shape[1])
        j = int(id_ % W.shape[1])
        if i == j:
            continue
        used.add(i)
        used.add(j)
        cnt += 1
        if cnt >= top_k:
            break
    N = W.shape[0]
    denom = float(min(N, 2 * top_k)) if N else 1.0
    return float(len(used) / denom)


def judge_formalization_quality(
    art: "PipelineArtifacts",
    *,
    # Convergence thresholds (already used in rollout, but we still judge)
    dx_good: float = 1e-4,
    dW_good: float = 1e-4,

    # W health (scale depends on eta/pruning; these are robust-ish)
    W_mass_min: float = 0.05,
    W_mass_max: float = 20.0,     # overgrown can indicate too-dense / too-fast updates
    W_density_min: float = 0.001, # too sparse => dead graph
    W_density_max: float = 0.25,  # too dense => mush

    # Activation health
    max_a_max: float = 0.98,      # >0.98 => one-node collapse likely
    eff_support_min: float = 3.0, # exp(entropy) < 3 => too concentrated

    # A stability
    A_norm_max: float = 1.0,      # should be <= 1 - contraction_eps, but allow tiny slack

    # Top-edge diversity
    top_edge_div_min: float = 0.10,
) -> QualityReport:
    """
    Automatic quality judge for the MVP formalization.

    It flags 3 common failure modes:
      - non-convergence / oscillation
      - dead graph (W ~ 0)
      - one-node collapse (a(x*) ~ one-hot) or mushy dense graph

    Returns:
      QualityReport(ok, verdict, metrics, flags, suggestions)
    """
    flags: List[str] = []
    suggestions: List[str] = []

    # Basic convergence metrics
    dx_last = float(art.result.history["dx"][-1]) if art.result.history.get("dx") else float("inf")
    dW_last = float(art.result.history["dW"][-1]) if art.result.history.get("dW") else float("inf")

    # Operator stability
    A_sn = float(spectral_norm(art.lower.A))

    # Graph stats
    W = art.result.W_star
    W_mass, W_max_edge, W_density = _edge_stats(W)

    # Activation stats
    a = _calc_activation_distribution(art)
    max_a = float(np.max(a)) if a.size else 1.0
    eff_supp = _effective_support(a) if a.size else 1.0
    ent = _entropy(a) if a.size else 0.0

    # Diversity
    div = _top_edge_diversity(art, top_k=50)

    metrics = {
        "converged": float(1.0 if art.result.converged else 0.0),
        "steps": float(art.result.steps),
        "dx_last": dx_last,
        "dW_last": dW_last,
        "A_spectral_norm": A_sn,
        "W_mass": W_mass,
        "W_max_edge": W_max_edge,
        "W_density": W_density,
        "a_max": max_a,
        "a_entropy": ent,
        "a_effective_support": eff_supp,
        "top_edge_diversity": div,
        "N_nodes": float(len(art.concept_nodes)),
    }

    # ---- Flags & suggestions ----

    # 1) Convergence / stability
    if not art.result.converged or dx_last > dx_good or dW_last > dW_good:
        flags.append("non_converged_or_weak_convergence")
        suggestions += [
            "contraction_eps를 올리세요 (예: 5e-3→1e-2).",
            "eta를 낮추세요 (예: 0.10→0.05).",
            "ridge_lambda를 올리세요 (예: 1e-2→5e-2).",
            "k_neighbors를 줄이세요 (예: 8→4~6).",
        ]

    if A_sn > A_norm_max + 1e-6:
        flags.append("A_not_contractive_enough")
        suggestions += [
            "A 수축 강제가 제대로 적용되는지 확인하세요(스펙트럴 노름 스케일링).",
            "contraction_eps를 더 크게 잡으세요.",
        ]

    # 2) Dead graph (W too small / too sparse)
    if W_mass < W_mass_min or W_density < W_density_min:
        flags.append("dead_or_too_sparse_graph")
        suggestions += [
            "prune_theta를 낮추세요 (예: 1e-4→1e-5).",
            "beta를 올리세요 (예: 12→16).",
            "eta를 약간 올리세요 (예: 0.10→0.15).",
            "노드 수 N을 늘리거나(예: 120→200), 코퍼스에서 노드별 컨텍스트 샘플이 충분한지 확인하세요.",
        ]

    # 3) Over-dense / mushy graph
    if W_mass > W_mass_max or W_density > W_density_max:
        flags.append("too_dense_or_mushy_graph")
        suggestions += [
            "prune_theta를 올리세요 (예: 1e-4→3e-4).",
            "eta를 낮추세요 (예: 0.10→0.05).",
            "beta를 낮추세요 (예: 12→8~10).",
        ]

    # 4) One-node collapse
    if max_a > max_a_max or eff_supp < eff_support_min:
        flags.append("activation_collapse_or_too_concentrated")
        suggestions += [
            "beta를 낮추세요 (예: 12→8~10).",
            "노드 사전에서 앵커/메타 노드 비율을 늘려보세요(특히 E, D).",
            "다의성이 큰 노드를 분해하거나(노드 분할), 너무 유사한 노드(중복)를 제거하세요.",
        ]

    # 5) Hub domination / low diversity
    if div < top_edge_div_min and W_mass >= W_mass_min:
        flags.append("low_top_edge_diversity_hub_dominated")
        suggestions += [
            "beta를 약간 낮추고(prune_theta는 조금 올려) 허브 집중을 줄이세요.",
            "Bucket C(대립/경계) 노드를 늘려 의미 경계를 강화하세요.",
            "노드 프로토타입(u_j) 품질을 높이세요(노드별 컨텍스트 샘플 수 증가).",
        ]

    # Deduplicate suggestions while preserving order
    seen = set()
    sug2 = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            sug2.append(s)
    suggestions = sug2

    # ---- Verdict ----
    if not flags:
        ok = True
        verdict = "OK: 수렴했고, W/활성 분포가 건강합니다."
    else:
        ok = False
        # prioritize by severity
        if "non_converged_or_weak_convergence" in flags:
            verdict = "NG: 수렴이 불충분합니다(안정화/수축/업데이트 속도 조정 필요)."
        elif "dead_or_too_sparse_graph" in flags:
            verdict = "NG: W가 죽어 있습니다(pruning/집중도/노드 품질 조정 필요)."
        elif "activation_collapse_or_too_concentrated" in flags:
            verdict = "NG: 활성 분포가 한 노드로 몰립니다(beta/노드 사전 조정 필요)."
        elif "too_dense_or_mushy_graph" in flags:
            verdict = "NG: W가 지나치게 조밀합니다(pruning/eta/beta 조정 필요)."
        else:
            verdict = "NG: 그래프 구조가 불안정합니다(다양성/노드/파라미터 조정 필요)."

    return QualityReport(
        ok=ok,
        verdict=verdict,
        metrics=metrics,
        flags=flags,
        suggestions=suggestions,
    )


# -------------------------
# Convenience printer
# -------------------------

def print_quality_report(q: QualityReport, top_metrics: Optional[List[str]] = None) -> None:
    top_metrics = top_metrics or [
        "converged", "steps", "dx_last", "dW_last",
        "A_spectral_norm",
        "W_mass", "W_density",
        "a_max", "a_effective_support",
        "top_edge_diversity",
        "N_nodes",
    ]
    print(q.verdict)
    print("Flags:", q.flags if q.flags else "[]")
    print("Key metrics:")
    for k in top_metrics:
        if k in q.metrics:
            v = q.metrics[k]
            # pretty formatting
            if abs(v) >= 1000 or (0 < abs(v) < 1e-3):
                print(f"  - {k}: {v:.6g}")
            else:
                print(f"  - {k}: {v:.6f}")
    if q.suggestions:
        print("Suggestions:")
        for s in q.suggestions:
            print("  -", s)


# -------------------------
# Example usage (after run_pipeline)
# -------------------------
if __name__ == "__main__":
    # Suppose you already ran:
    # art = run_pipeline(...)
    #
    # Then:
    # q = judge_formalization_quality(art)
    # print_quality_report(q)
    pass
```

### 기준을 “더 빡세게/더 느슨하게” 만들고 싶을 때(정확히 어디를 건드리면 되는지)

- **쓸모없는 수렴(다 0)**을 더 빨리 잡고 싶으면: `W_mass_min`, `W_density_min`을 올리세요.
    
- **한 노드 몰림**을 더 빨리 잡고 싶으면: `max_a_max`를 0.95로 내리거나 `eff_support_min`을 5로 올리세요.
    
- **그래프가 너무 조밀한 것**을 더 빨리 잡고 싶으면: `W_density_max`를 0.15로 내리세요.
    

원하시면, 이 판정 함수가 반환한 `flags`를 입력으로 받아서 **cfg를 자동으로 업데이트하는 `auto_tune_cfg()`**까지 이어서 만들어드릴까요?

# auto_tune_cfg() 함수
```python
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Optional
import math


# Assumes you already have:
# - SRConfig (from earlier)
# - NodeDictConfig (from earlier)
# - QualityReport (from judge_formalization_quality)
#
# This module adds:
# - auto_tune_cfg(): returns new (sr_cfg, node_cfg) based on flags/metrics
# - apply_auto_tune_loop(): optional helper to iterate tune->run->judge cycles


@dataclass
class TuneStep:
    key: str
    old: float
    new: float
    reason: str


@dataclass
class TuneResult:
    sr_cfg_new: "SRConfig"
    node_cfg_new: "NodeDictConfig"
    steps: List[TuneStep]
    notes: List[str]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def auto_tune_cfg(
    q: "QualityReport",
    sr_cfg: "SRConfig",
    node_cfg: "NodeDictConfig",
    *,
    # safety clamps (so you don't blow up runs)
    eta_range: Tuple[float, float] = (0.02, 0.30),
    beta_range: Tuple[float, float] = (4.0, 30.0),
    prune_theta_range: Tuple[float, float] = (1e-7, 1e-2),
    ridge_lambda_range: Tuple[float, float] = (1e-4, 5e-1),
    contraction_eps_range: Tuple[float, float] = (1e-4, 5e-2),
    k_neighbors_range: Tuple[int, int] = (2, 12),
    k_contexts_range: Tuple[int, int] = (200, 5000),
    N_range: Tuple[int, int] = (40, 400),
) -> TuneResult:
    """
    Deterministic auto-tuner that converts judge flags/metrics into SRConfig/NodeDictConfig updates.

    Philosophy:
      - Fix stability first (contraction/eta/lambda).
      - Then fix dead graph (prune/beta/eta/N).
      - Then fix collapse/density/diversity.

    Returns a *new* config pair; does not mutate inputs.
    """
    steps: List[TuneStep] = []
    notes: List[str] = []

    # Make mutable copies via dataclasses.replace
    sr = sr_cfg
    nd = node_cfg

    flags = set(q.flags)
    m = q.metrics

    def set_sr(key: str, value, reason: str):
        nonlocal sr, steps
        old = getattr(sr, key)
        if old == value:
            return
        sr = replace(sr, **{key: value})
        # store as float when possible
        try:
            oldf = float(old)
            newf = float(value)
            steps.append(TuneStep(key=f"sr.{key}", old=oldf, new=newf, reason=reason))
        except Exception:
            steps.append(TuneStep(key=f"sr.{key}", old=float("nan"), new=float("nan"), reason=f"{reason} (non-float)"))

    def set_nd(key: str, value, reason: str):
        nonlocal nd, steps
        old = getattr(nd, key)
        if old == value:
            return
        nd = replace(nd, **{key: value})
        steps.append(TuneStep(key=f"node.{key}", old=float(old), new=float(value), reason=reason))

    # Helper getters
    eta = float(sr.eta)
    beta = float(sr.beta)
    theta = float(sr.prune_theta)
    lam = float(sr.ridge_lambda)
    eps = float(sr.contraction_eps)
    kN = int(sr.k_neighbors)
    kC = int(sr.k_contexts)
    N = int(nd.N)

    # ------------------------------------------------------------------
    # 1) Stabilize if non-converged / A not contractive
    # ------------------------------------------------------------------
    if "non_converged_or_weak_convergence" in flags or "A_not_contractive_enough" in flags:
        # Strongest levers first:
        eps2 = _clamp(eps * 2.0, *contraction_eps_range)
        eta2 = _clamp(eta * 0.70, *eta_range)
        lam2 = _clamp(lam * 5.0, *ridge_lambda_range)
        kN2 = int(_clamp(kN - 2, k_neighbors_range[0], k_neighbors_range[1]))

        set_sr("contraction_eps", eps2, "stabilize: enforce stronger contraction")
        set_sr("eta", eta2, "stabilize: slow upper-layer updates")
        set_sr("ridge_lambda", lam2, "stabilize: stronger ridge regularization for A,b fit")
        set_sr("k_neighbors", kN2, "stabilize: reduce noisy next-context averaging")

    # ------------------------------------------------------------------
    # 2) Dead/sparse graph: W too small / too sparse
    # ------------------------------------------------------------------
    if "dead_or_too_sparse_graph" in flags:
        # Main levers: lower pruning threshold, increase concentration, increase update rate, increase node coverage
        theta2 = _clamp(theta * 0.10, *prune_theta_range)
        beta2 = _clamp(beta + 2.0, *beta_range)
        eta2 = _clamp(eta * 1.25, *eta_range)

        # increase N if not already large
        N2 = int(_clamp(int(round(N * 1.5)), N_range[0], N_range[1]))

        set_sr("prune_theta", theta2, "revive W: prune less aggressively")
        set_sr("beta", beta2, "revive W: sharpen a(x) so S(x) has stronger edges")
        set_sr("eta", eta2, "revive W: let W grow faster")
        if N2 > N:
            set_nd("N", N2, "revive W: enlarge node dictionary to provide more structure")

    # ------------------------------------------------------------------
    # 3) Too dense / mushy graph
    # ------------------------------------------------------------------
    if "too_dense_or_mushy_graph" in flags:
        theta2 = _clamp(theta * 3.0, *prune_theta_range)
        eta2 = _clamp(eta * 0.70, *eta_range)
        beta2 = _clamp(beta - 2.0, *beta_range)

        set_sr("prune_theta", theta2, "de-mush: prune weak edges more")
        set_sr("eta", eta2, "de-mush: slow W growth")
        set_sr("beta", beta2, "de-mush: soften a(x) to avoid uniform dense S(x)")

    # ------------------------------------------------------------------
    # 4) Activation collapse (one-node dominance)
    # ------------------------------------------------------------------
    if "activation_collapse_or_too_concentrated" in flags:
        beta2 = _clamp(beta - 2.0, *beta_range)

        # Increase anchors/meta share by increasing N a bit (more room),
        # but without changing bucket ratios here (you already fixed the rule).
        # If you want ratio-level tuning, we can add it, but keeping it minimal.
        N2 = int(_clamp(int(round(N * 1.2)), N_range[0], N_range[1]))

        set_sr("beta", beta2, "reduce collapse: soften a(x)")
        if N2 > N:
            set_nd("N", N2, "reduce collapse: more nodes give more competing prototypes")

    # ------------------------------------------------------------------
    # 5) Low diversity among top edges (hub-dominated)
    # ------------------------------------------------------------------
    if "low_top_edge_diversity_hub_dominated" in flags:
        # Slightly reduce beta; slightly increase pruning to cut hub's weak spokes.
        beta2 = _clamp(beta - 1.0, *beta_range)
        theta2 = _clamp(theta * 1.5, *prune_theta_range)

        # Also consider increasing contrast bucket indirectly by increasing N,
        # because your node builder fills buckets by fixed ratios.
        N2 = int(_clamp(int(round(N * 1.25)), N_range[0], N_range[1]))

        set_sr("beta", beta2, "increase diversity: soften a(x) to avoid single hub")
        set_sr("prune_theta", theta2, "increase diversity: prune weak hub spokes")
        if N2 > N:
            set_nd("N", N2, "increase diversity: larger dictionary adds boundary/contrast nodes via fixed ratios")

    # ------------------------------------------------------------------
    # 6) If things look OK, do nothing (or optionally tighten for efficiency)
    # ------------------------------------------------------------------
    if not q.flags:
        notes.append("Quality OK: no auto-tuning applied.")
        return TuneResult(sr_cfg_new=sr, node_cfg_new=nd, steps=steps, notes=notes)

    # ------------------------------------------------------------------
    # 7) Optional: coarse adjustments using metrics (when flags ambiguous)
    # ------------------------------------------------------------------
    # If W is *almost* dead but not flagged due to thresholds, nudge theta down a bit
    if m.get("W_mass", 1.0) < 0.10 and "dead_or_too_sparse_graph" not in flags:
        theta2 = _clamp(float(sr.prune_theta) * 0.5, *prune_theta_range)
        set_sr("prune_theta", theta2, "metric nudge: low W_mass, prune slightly less")

    # If a_max is extremely high, push beta down more strongly
    if m.get("a_max", 0.0) > 0.995:
        beta2 = _clamp(float(sr.beta) - 3.0, *beta_range)
        set_sr("beta", beta2, "metric nudge: extreme one-hot a(x), reduce beta more")

    # If A spectral norm too close to 1, strengthen contraction a bit
    if m.get("A_spectral_norm", 0.0) > 0.999:
        eps2 = _clamp(float(sr.contraction_eps) * 1.5, *contraction_eps_range)
        set_sr("contraction_eps", eps2, "metric nudge: A norm ~1, strengthen contraction")

    # Final clamp sanity (in case of user-edited cfg outside range)
    set_sr("eta", _clamp(float(sr.eta), *eta_range), "sanity clamp: eta")
    set_sr("beta", _clamp(float(sr.beta), *beta_range), "sanity clamp: beta")
    set_sr("prune_theta", _clamp(float(sr.prune_theta), *prune_theta_range), "sanity clamp: prune_theta")
    set_sr("ridge_lambda", _clamp(float(sr.ridge_lambda), *ridge_lambda_range), "sanity clamp: ridge_lambda")
    set_sr("contraction_eps", _clamp(float(sr.contraction_eps), *contraction_eps_range), "sanity clamp: contraction_eps")
    set_sr("k_neighbors", int(_clamp(int(sr.k_neighbors), k_neighbors_range[0], k_neighbors_range[1])), "sanity clamp: k_neighbors")
    set_sr("k_contexts", int(_clamp(int(sr.k_contexts), k_contexts_range[0], k_contexts_range[1])), "sanity clamp: k_contexts")
    set_nd("N", int(_clamp(int(nd.N), N_range[0], N_range[1])), "sanity clamp: N")

    # Deduplicate step list for repeated set of same key (keep last change)
    # (Simple method: keep only last by key)
    last_by_key: Dict[str, TuneStep] = {}
    for st in steps:
        last_by_key[st.key] = st
    steps2 = list(last_by_key.values())
    steps2.sort(key=lambda s: s.key)
    steps = steps2

    notes.append(f"Applied auto-tuning for flags: {sorted(list(flags))}")
    return TuneResult(sr_cfg_new=sr, node_cfg_new=nd, steps=steps, notes=notes)


# ----------------------------------------------------------------------
# Optional helper: one-stop loop (you call run_pipeline yourself inside)
# ----------------------------------------------------------------------

@dataclass
class AutoTuneLoopResult:
    artifacts_list: List["PipelineArtifacts"]
    reports: List["QualityReport"]
    tune_steps: List[TuneResult]


def apply_auto_tune_loop(
    run_pipeline_fn,
    judge_fn,
    concept: str,
    corpus_docs,
    embed,
    sr_cfg: "SRConfig",
    node_cfg: "NodeDictConfig",
    *,
    max_rounds: int = 4,
    seed: int = 7,
) -> AutoTuneLoopResult:
    """
    Convenience loop:
      for round in 1..max_rounds:
        art = run_pipeline(...)
        q = judge(...)
        if q.ok: break
        (sr_cfg, node_cfg) = auto_tune_cfg(q,...)

    You supply:
      - run_pipeline_fn: function like run_pipeline(...)
      - judge_fn: function like judge_formalization_quality(...)

    Returns full trace for analysis.
    """
    artifacts_list = []
    reports = []
    tune_results = []

    sr = sr_cfg
    nd = node_cfg

    for r in range(1, max_rounds + 1):
        art = run_pipeline_fn(concept=concept, corpus_docs=corpus_docs, embed=embed, sr_cfg=sr, node_cfg=nd, seed=seed)
        q = judge_fn(art)

        artifacts_list.append(art)
        reports.append(q)

        if q.ok:
            break

        tr = auto_tune_cfg(q, sr, nd)
        tune_results.append(tr)
        sr, nd = tr.sr_cfg_new, tr.node_cfg_new

    return AutoTuneLoopResult(
        artifacts_list=artifacts_list,
        reports=reports,
        tune_steps=tune_results,
    )


# ----------------------------------------------------------------------
# Minimal printer
# ----------------------------------------------------------------------

def print_tune_result(tr: TuneResult) -> None:
    print("Auto-tune notes:")
    for n in tr.notes:
        print("  -", n)
    if not tr.steps:
        print("No parameter changes.")
        return
    print("Changed parameters:")
    for s in tr.steps:
        # pretty formatting for tiny numbers
        old = f"{s.old:.6g}" if (abs(s.old) < 1e-3 or abs(s.old) >= 1e3) else f"{s.old:.6f}"
        new = f"{s.new:.6g}" if (abs(s.new) < 1e-3 or abs(s.new) >= 1e3) else f"{s.new:.6f}"
        print(f"  - {s.key}: {old} -> {new}  ({s.reason})")
```

원하시는 흐름은 이렇게 됩니다.

1. `art = run_pipeline(...)`
    
2. `q = judge_formalization_quality(art)`
    
3. `tr = auto_tune_cfg(q, sr_cfg, node_cfg)`
    
4. `sr_cfg, node_cfg = tr.sr_cfg_new, tr.node_cfg_new`
    
5. 다시 1)로 반복
    

그리고 귀찮으면 `apply_auto_tune_loop(run_pipeline, judge_formalization_quality, ...)`로 한 번에 돌리시면 됩니다.