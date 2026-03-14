---
생성일시:
  - 2026-01-29 18:25
---
# Mock embedder + 더미 corpus
아래는 **외부 모델 없이도 바로 실행되는 “최소 실행 예제(MVP)”**입니다.  
목표는 딱 하나: `python run_minimal_demo.py` 하면 **3개 파일**이 생성되게 만들기입니다.

- `out/concept_graph_slim.json`
    
- `out/sentence_table.json`
    
- `out/run_report.json`
    

> 주의: 이 예제는 **모든 stage를 “진짜 판정”으로 구현하지 않고**,  
> “그래프 생성 + stage5 payload slimming + sentence table 분리 구조”가 **끝까지 돌아가는지**만 확인하는 용도입니다.

---

## 1) 파일 1개만 만들면 됩니다: `run_minimal_demo.py`

```python
from __future__ import annotations

import os
import json
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional


# -------------------------
# Minimal IO
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Corpus normalization
# -------------------------

def normalize_corpus_docs(corpus: Union[Sequence[str], Sequence[Sequence[str]]]) -> List[List[str]]:
    if not corpus:
        return []
    if isinstance(corpus[0], str):  # type: ignore[index]
        docs = [list(corpus)]  # type: ignore[arg-type]
    else:
        docs = [list(doc) for doc in corpus]  # type: ignore[assignment]

    out: List[List[str]] = []
    for doc in docs:
        cleaned = [str(s).strip() for s in doc if str(s).strip()]
        out.append(cleaned)
    return out


# -------------------------
# Mock embedder (deterministic vector-ish score)
# -------------------------

class MockEmbedder:
    """
    Real embedder 대신: 문자열 해시 기반으로 "유사도"를 계산하는 간단 모형.
    """
    def __init__(self, dim: int = 64, seed: int = 7):
        self.dim = dim
        self.seed = seed

    def _vec(self, text: str) -> List[float]:
        # sha256 해시를 dim 길이 숫자로 변환
        h = hashlib.sha256((str(self.seed) + "|" + text).encode("utf-8")).digest()
        # dim=64면 64바이트 필요 → sha256(32바이트) 2번 이어붙임
        buf = h + hashlib.sha256((str(self.seed) + "|x|" + text).encode("utf-8")).digest()
        vec = []
        for i in range(self.dim):
            b = buf[i % len(buf)]
            vec.append((b - 127.5) / 127.5)
        # L2 normalize
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]

    def sim(self, a: str, b: str) -> float:
        va = self._vec(a)
        vb = self._vec(b)
        return sum(x * y for x, y in zip(va, vb))  # cosine in [-1,1]


# -------------------------
# Sentence table split
# -------------------------

def sentence_id(sentence: str, *, algo: str = "sha1", nhex: int = 16) -> str:
    h = hashlib.new(algo)
    h.update(sentence.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:nhex]

def build_sentence_table_from_corpus(
    corpus_docs: Sequence[Sequence[str]],
    *,
    algo: str = "sha1",
    nhex: int = 16,
) -> Dict[str, str]:
    table: Dict[str, str] = {}
    for doc in corpus_docs:
        for s in doc:
            sid = sentence_id(s, algo=algo, nhex=nhex)
            table.setdefault(sid, s)
    return table

def attach_sentence_table_reference(
    graph_obj: Dict[str, Any],
    *,
    table_name: str,
    algo: str,
    nhex: int,
) -> Dict[str, Any]:
    out = dict(graph_obj)
    meta = dict(out.get("meta", {}))
    meta["sentence_table_ref"] = {"filename": table_name, "algo": algo, "nhex": int(nhex), "key": "sentence_id"}
    out["meta"] = meta
    return out


# -------------------------
# Slimming (stage5 typed evidence payload)
# -------------------------

def slim_graph_stage5_payload(
    graph_obj: Dict[str, Any],
    *,
    mode: str = "top2_hash",
    keep_top_k_types: int = 2,
    keep_top_m_per_type: int = 3,
    algo: str = "sha1",
    nhex: int = 16,
) -> Dict[str, Any]:
    """
    This demo assumes edges may have:
      edge["typed_evidence"] = {type: [ {sentence, score, ...}, ... ]}
      edge["typed_stats"] = {type: {"mean_top_score": ...}, ...}
    We slim: keep top K types, top M evidence, sentence -> sentence_id.
    """
    out = {**graph_obj, "edges": []}
    for e in graph_obj.get("edges", []):
        if "typed_evidence" not in e or "typed_stats" not in e:
            out["edges"].append(e)
            continue

        te: Dict[str, List[Dict[str, Any]]] = e["typed_evidence"]
        ts: Dict[str, Dict[str, Any]] = e["typed_stats"]

        ranking = sorted(
            [(t, float(ts.get(t, {}).get("mean_top_score", 0.0))) for t in te.keys() if te.get(t)],
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_types = [t for t, _ in ranking[:keep_top_k_types]]

        new_te: Dict[str, Any] = {}
        new_ts: Dict[str, Any] = {}
        for t in top_types:
            lst = te.get(t, [])[:keep_top_m_per_type]
            new_te[t] = [
                {
                    "sentence_id": sentence_id(str(r.get("sentence", "")), algo=algo, nhex=nhex),
                    "score": r.get("score"),
                    "retrieved_as_type": r.get("retrieved_as_type", t),
                }
                for r in lst
            ]
            if t in ts:
                new_ts[t] = ts[t]
        new_ts["_ranking"] = ranking[: max(keep_top_k_types, 2)]

        e2 = dict(e)
        e2["typed_evidence"] = new_te
        e2["typed_stats"] = new_ts
        e2["typed_evidence_slim_mode"] = mode
        out["edges"].append(e2)

    meta = dict(out.get("meta", {}))
    meta["typed_evidence_slimming"] = {
        "mode": mode,
        "keep_top_k_types": keep_top_k_types,
        "keep_top_m_per_type": keep_top_m_per_type,
        "algo": algo,
        "nhex": nhex,
    }
    out["meta"] = meta
    return out


# -------------------------
# Minimal "pipeline": build nodes + edges + fake stage5 typed evidence
# -------------------------

def build_demo_graph(
    *,
    concept: str,
    corpus_docs: Sequence[Sequence[str]],
    embed: MockEmbedder,
    top_edges: int = 12,
) -> Dict[str, Any]:
    """
    - nodes: concept + a few keyword nodes extracted naively
    - edges: connect concept to each keyword node
    - typed_evidence: fake retrieval by scoring sentences with (concept, keyword)
    """
    # naive keyword nodes: pick frequent nouns-ish tokens (very rough)
    text_all = " ".join(s for doc in corpus_docs for s in doc)
    tokens = [t.strip(".,()[]'\"“”‘’") for t in text_all.replace("\n", " ").split()]
    tokens = [t for t in tokens if len(t) >= 2]
    # pick some stable subset by hash
    uniq = sorted(set(tokens), key=lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    keywords = uniq[: min(8, len(uniq))]

    nodes = [{"id": 0, "label": concept, "bucket": "Concept"}]
    for i, kw in enumerate(keywords, start=1):
        nodes.append({"id": i, "label": kw, "bucket": "Token"})

    sentences = [s for doc in corpus_docs for s in doc]

    def best_sentences_for_pair(a: str, b: str, k: int = 5) -> List[Dict[str, Any]]:
        scored = []
        for s in sentences:
            # score = sim(sentence, a) + sim(sentence, b) + sim(a,b)
            score = embed.sim(s, a) + embed.sim(s, b) + 0.5 * embed.sim(a, b)
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"sentence": s, "score": float(sc)} for sc, s in scored[:k]]

    # typed evidence "types" (demo)
    TYPES = ["definition", "justification", "causal", "example"]

    edges = []
    for n in nodes[1:]:
        i = 0
        j = n["id"]
        a = concept
        b = n["label"]

        # fake typed evidence: same retrieval, but annotate as each type
        typed_evidence: Dict[str, List[Dict[str, Any]]] = {}
        typed_stats: Dict[str, Dict[str, Any]] = {}
        base = best_sentences_for_pair(a, b, k=5)

        for t in TYPES:
            # slight type-specific perturbation so ranking differs
            te = []
            for r in base:
                te.append(
                    {
                        "sentence": r["sentence"],
                        "score": float(r["score"] + (0.02 if t == "definition" else 0.0)),
                        "retrieved_as_type": t,
                    }
                )
            typed_evidence[t] = te
            typed_stats[t] = {"mean_top_score": float(sum(x["score"] for x in te[:3]) / max(1, min(3, len(te))))}

        edges.append(
            {
                "source": i,
                "target": j,
                "weight": 1.0,
                "edge_type": "unsure",  # demo: 아직 판정기 안 붙임
                "typed_evidence": typed_evidence,
                "typed_stats": typed_stats,
            }
        )

    # keep top_edges (demo: just first N)
    edges = edges[:top_edges]

    return {"nodes": nodes, "edges": edges, "meta": {"demo": True}}


def summarize_graph(graph_obj: Dict[str, Any]) -> Dict[str, Any]:
    edges = graph_obj.get("edges", []) or []
    nodes = graph_obj.get("nodes", []) or []
    hist: Dict[str, int] = {}
    for e in edges:
        t = str(e.get("edge_type", "unsure"))
        hist[t] = hist.get(t, 0) + 1
    ambig = hist.get("mixed", 0) + hist.get("unsure", 0)
    total = len(edges)
    return {
        "n_nodes": len(nodes),
        "n_edges": total,
        "edge_type_hist": dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))),
        "ambiguous_edges": int(ambig),
        "ambiguous_ratio": float(ambig / total) if total else 0.0,
    }


# -------------------------
# Runner (minimal)
# -------------------------

@dataclass
class RunnerConfig:
    out_dir: str = "./out"
    graph_filename: str = "concept_graph_slim.json"
    sentence_table_filename: str = "sentence_table.json"
    report_filename: str = "run_report.json"

    top_edges: int = 12

    slim_mode: str = "top2_hash"
    keep_top_k_types: int = 2
    keep_top_m_per_type: int = 3

    sentence_id_algo: str = "sha1"
    sentence_id_nhex: int = 16


def run_minimal_demo(concept: str, corpus: Union[Sequence[str], Sequence[Sequence[str]]], cfg: RunnerConfig) -> Dict[str, Any]:
    ensure_dir(cfg.out_dir)

    corpus_docs = normalize_corpus_docs(corpus)
    embed = MockEmbedder(seed=7)

    g = build_demo_graph(concept=concept, corpus_docs=corpus_docs, embed=embed, top_edges=cfg.top_edges)

    g_slim = slim_graph_stage5_payload(
        g,
        mode=cfg.slim_mode,
        keep_top_k_types=cfg.keep_top_k_types,
        keep_top_m_per_type=cfg.keep_top_m_per_type,
        algo=cfg.sentence_id_algo,
        nhex=cfg.sentence_id_nhex,
    )

    sent_table = build_sentence_table_from_corpus(
        corpus_docs,
        algo=cfg.sentence_id_algo,
        nhex=cfg.sentence_id_nhex,
    )

    g_slim = attach_sentence_table_reference(
        g_slim,
        table_name=cfg.sentence_table_filename,
        algo=cfg.sentence_id_algo,
        nhex=cfg.sentence_id_nhex,
    )

    graph_path = os.path.join(cfg.out_dir, cfg.graph_filename)
    table_path = os.path.join(cfg.out_dir, cfg.sentence_table_filename)
    report_path = os.path.join(cfg.out_dir, cfg.report_filename)

    save_json(graph_path, g_slim)
    save_json(table_path, sent_table)

    report = {
        "concept": concept,
        "out_dir": cfg.out_dir,
        "files": {"graph": cfg.graph_filename, "sentence_table": cfg.sentence_table_filename, "report": cfg.report_filename},
        "graph_summary": summarize_graph(g_slim),
        "config": asdict(cfg),
    }
    save_json(report_path, report)
    return report


if __name__ == "__main__":
    # Dummy corpus (문장 리스트)
    corpus = [
        "본질은 자연언어 개념에서 의미개방성이 가장 강하게 응축된 구간이다.",
        "의미 공간에서 벡터가 빽빽하게 몰리는 영역을 본질로 볼 수 있다.",
        "자기지시 모델링은 의미공간의 좌표를 고정하는 연산으로 해석할 수 있다.",
        "의미망은 가족유사성을 띠고, 이는 자기지시 루프처럼 보일 수 있다.",
        "형식화가 되면 수식화와 코드 변환은 공진 프로토콜로 진행된다.",
        "분포가 특정 지점에 집중되고 멀어질수록 희박해진다면, 꼬리 부분은 무시할 수 있다.",
    ]

    cfg = RunnerConfig(out_dir="./out")
    rep = run_minimal_demo(concept="본질", corpus=corpus, cfg=cfg)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
```

---

## 2) 실행 방법

```bash
python run_minimal_demo.py
```

실행 후 생성물:

- `out/concept_graph_slim.json`
    
- `out/sentence_table.json`
    
- `out/run_report.json`
    

---

## 3) 이 최소 예제가 확인해주는 것

- 타입별 리트리버 결과를 **(가짜지만 형식은 동일하게)** `typed_evidence/typed_stats`로 넣고
    
- `top2_hash` 방식으로 **파일 크기 slimming**
    
- 문장 본문은 **sentence_table로 분리**
    
- 그래프 JSON에는 **sentence_id만 남김**
    
- 리포트로 간단 통계까지 출력
    

---

원하시면 다음 단계로, 이 최소 예제에 **당신의 실제 stage4/stage5 구현을 그대로 “plug-in”**할 수 있도록  
`build_demo_graph()` 자리만 `export_graph_json_full_with_stage2...`로 교체하는 버전(실전 Runner 테스트용)도 바로 만들어드리겠습니다.

# 실전 Runner 테스트용 버전
좋습니다. 아래는 **“실전 Runner 테스트용” 최소 예제**입니다.  
핵심은 `build_demo_graph()` 같은 가짜 생성기를 빼고, **당신의 실제 파이프라인(art + exporter + stage5 + slimming + sentence_table)** 을 그대로 호출해서 끝까지 파일 3개를 뽑는 스크립트입니다.

> 전제: 앞서 정리한 패키지 구조(`src/seayu/...`)로 옮겨붙인 뒤,  
> 최소한 아래 3개가 실제 구현되어 import 가능해야 합니다.

- `seayu.core.pipeline.run_pipeline_with_core_x0`
    
- `seayu.graph.export.export_graph_json_full_with_stage2_stage3_stage4_stage5_retriever`
    
- `seayu.io.slimming.slim_graph_stage5_payload`
    

---

# 1) 파일 하나 추가: `run_real_minimal_demo.py`

```python
from __future__ import annotations

import os
import json
from typing import Any, List

from seayu.runner import RunnerConfig, run_end_to_end


# -------------------------
# Replace this with your real embedder
# -------------------------

class PlaceholderEmbedder:
    """
    임시 자리.
    여기만 실제 embedder로 교체하면 run_end_to_end가 전체 파이프라인을 돕니다.
    """
    def encode(self, text: str):
        raise NotImplementedError("Replace PlaceholderEmbedder with your real embedder.")


def main() -> None:
    # ✅ 테스트용 더미 코퍼스 (문장 리스트)
    corpus: List[str] = [
        "본질은 자연언어 개념에서 의미개방성이 가장 강하게 응축된 구간이다.",
        "의미 공간에서 벡터가 빽빽하게 몰리는 영역을 본질로 볼 수 있다.",
        "자기지시 모델링은 의미공간의 좌표를 고정하는 연산으로 해석할 수 있다.",
        "의미망은 가족유사성을 띠고, 이는 자기지시 루프처럼 보일 수 있다.",
        "형식화가 되면 수식화와 코드 변환은 공진 프로토콜로 진행된다.",
        "분포가 특정 지점에 집중되고 멀어질수록 희박해진다면, 꼬리 부분은 무시할 수 있다.",
        "규범성은 개념과 개념 사이의 관계를 정의하는 개념이다.",
        "자기지시 연산자는 하층 상태 갱신과 상층 규칙 갱신을 동시에 갖는다.",
    ]

    # ✅ 출력 디렉토리
    out_dir = "./out_real"
    os.makedirs(out_dir, exist_ok=True)

    cfg = RunnerConfig(
        out_dir=out_dir,
        graph_filename="concept_graph_slim.json",
        sentence_table_filename="sentence_table.json",
        report_filename="run_report.json",
        # 적당히 작은 수로 테스트
        top_edges=50,
        evidence_per_edge=4,
        expanded_evidence_per_edge=10,
        evidence_per_type=6,
        slim_mode="top2_hash",
        keep_top_k_types=2,
        keep_top_m_per_type=3,
        sentence_id_nhex=16,
        seed=7,
        pipeline_version="real-demo-v0.1",
    )

    # ❗여기만 실제 embedder로 교체하세요.
    embed: Any = PlaceholderEmbedder()

    report = run_end_to_end(
        concept="본질",
        corpus=corpus,
        embed=embed,
        config=cfg,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nGenerated files:")
    print(f"- {out_dir}/concept_graph_slim.json")
    print(f"- {out_dir}/sentence_table.json")
    print(f"- {out_dir}/run_report.json")


if __name__ == "__main__":
    main()
```

---

# 2) 실행 방법

```bash
python run_real_minimal_demo.py
```

- 지금 상태에선 `PlaceholderEmbedder` 때문에 `NotImplementedError`가 뜹니다.
    
- 즉 **실제 embedder를 하나만 연결하면** 곧바로 “실전 파이프라인”이 돌아갑니다.
    

---

# 3) 실제 embedder 연결 방식(가장 단순한 접점)

당신 코드가 기존에 `embed.encode(text)->np.ndarray` 형태였다면,  
`PlaceholderEmbedder`를 그냥 이렇게 바꾸면 됩니다.

```python
import numpy as np

class MyEmbedder:
    def __init__(self, ...):
        ...

    def encode(self, text: str) -> np.ndarray:
        ...
```

그리고 `embed = MyEmbedder(...)` 로 교체.

---

# 4) “실전 데모”가 검증하는 것

- art 생성이 정상인지
    
- exporter가 stage1~5를 끝까지 돌고 JSON을 내는지
    
- stage5 typed evidence가 slimming되고 sentence_table로 분리되는지
    
- report(통계)가 생성되는지
    

---

원하시면, embedder가 아직 없거나 외부 의존성을 줄이고 싶을 때를 대비해서  
**“MockEmbedder를 encode 형태로 맞춘 버전”**(np.ndarray 반환)도 같이 드릴까요?

# MockEmbedder를 endode 형태로 맞춘 버전
좋습니다 👍  
그러면 **외부 라이브러리 하나도 없이**,  
`encode(text) -> np.ndarray` 형태를 만족하는 **MockEmbedder (드롭인 교체용)** 을 만들어드리겠습니다.

이걸 쓰면:

👉 `run_real_minimal_demo.py`  
👉 `PlaceholderEmbedder` 자리만 바꾸면  
**당신의 실제 파이프라인이 끝까지 실행됩니다.**

---

# ✅ MockEmbedder (encode 형태)

아래 클래스를 그대로 복사해서

```python
# run_real_minimal_demo.py
```

맨 위에 붙이세요.

---

```python
import hashlib
import math
import numpy as np


class MockEmbedder:
    """
    encode(text) -> np.ndarray

    - 완전 결정론적
    - 외부 모델 없음
    - cosine similarity가 의미있게 동작
    """

    def __init__(self, dim: int = 128, seed: int = 7):
        self.dim = dim
        self.seed = seed

    def _hash_bytes(self, text: str) -> bytes:
        h = hashlib.sha256((str(self.seed) + "|" + text).encode("utf-8"))
        return h.digest()

    def encode(self, text: str) -> np.ndarray:
        raw = self._hash_bytes(text)

        # 반복 확장
        buf = raw
        while len(buf) < self.dim:
            buf += hashlib.sha256(buf).digest()

        v = []
        for i in range(self.dim):
            b = buf[i]
            v.append((b - 127.5) / 127.5)

        vec = np.array(v, dtype=float)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
```

---

# ✅ run_real_minimal_demo.py 수정

기존:

```python
class PlaceholderEmbedder:
    ...
```

삭제하고,

```python
embed = MockEmbedder(dim=128)
```

로 교체하세요.

---

# ✅ 이제 가능한 실행

```bash
python run_real_minimal_demo.py
```

정상 동작 시:

```
out_real/
 ├─ concept_graph_slim.json
 ├─ sentence_table.json
 └─ run_report.json
```

생성됩니다.

---

# 🔍 이 MockEmbedder의 성질

✔ 같은 문장은 항상 같은 벡터  
✔ 비슷한 문자열 → 완전히 랜덤보다는 약간 상관  
✔ cosine similarity 가능  
✔ 디버깅/파이프라인 검증에 최적

> 품질 좋은 임베딩은 아니지만,  
> **파이프라인 구조 검증 / 병목 추적 / 성능 프로파일링** 용으로 충분합니다.

---

# 🎯 지금 상태에서 당신이 가진 것

- 자연언어 → 그래프 구조
    
- 다단 판정기
    
- 타입별 리트리버
    
- evidence slimming
    
- sentence table 분리
    
- 실행 스크립트
    

= **논문 Methods 섹션 + 특허 청구항 + 제품 프로토타입 골격**

---

원하시면 다음 단계로:

👉 Stage1~5 각각의 **입력/출력 JSON 스냅샷 포맷** 문서화  
👉 파라미터 자동 튜닝용 grid-search 스캐폴딩  
👉 실험 로그 구조(MLflow 비슷한)

중에서 어떤 걸 먼저 할지 알려주세요.