#!/usr/bin/env python3
"""Daily paper digest for microglial morphology–molecular state research."""

import os, json, time, smtplib, re, html, requests, feedparser
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import quote
from openai import OpenAI

# ── Search Config ────────────────────────────────────────────────────────────

ARXIV_QUERIES = [
    "microglia morphology computational image analysis",
    "microglia gene expression spatial transcriptomics",
    "morphology gene expression multimodal integration",
    "cell morphology representation learning microscopy",
    "spatial omics multimodal deep learning",
    "biomedical image encoder cell morphology",
    "graph neural network cell morphology skeleton",
    "unpaired single cell multimodal integration optimal transport contrastive learning",
]
ARXIV_CATS = (
    "cat:cs.LG OR cat:stat.ML OR cat:stat.ME "
    "OR cat:q-bio.QM OR cat:q-bio.GN OR cat:cs.AI"
)

# Classic queries: foundational topics for high-citation older papers
SS_CLASSIC_QUERIES = [
    "microglia morphology activation ramification Alzheimer disease",
    "cell morphology gene expression integration",
    "Patch-seq morphology transcriptomics neural cells",
    "spatial transcriptomics computational methods",
    "multimodal single cell integration CCA optimal transport contrastive learning",
]

PUBMED_QUERIES = [
    "microglia morphology gene expression",
    "microglia spatial transcriptomics Alzheimer disease",
    "cell morphology transcriptomics multimodal integration",
    "microscopy image representation learning cell morphology",
    "Patch-seq morphology transcriptomics",
]

# Top journals relevant to microglia, spatial omics, and computational imaging
JOURNAL_FILTER = (
    '"Nature"[Journal] OR "Nature Methods"[Journal] OR "Nature Biotechnology"[Journal] OR '
    '"Nature Medicine"[Journal] OR "Nature Communications"[Journal] OR '
    '"Cell"[Journal] OR "Cell Systems"[Journal] OR "Cell Genomics"[Journal] OR '
    '"Science"[Journal] OR "Science Translational Medicine"[Journal] OR '
    '"The New England Journal of Medicine"[Journal] OR "Lancet"[Journal] OR '
    '"JAMA"[Journal] OR "Bioinformatics"[Journal] OR "Genome Biology"[Journal] OR '
    '"Genome Research"[Journal] OR "Journal of Machine Learning Research"[Journal]'
)
JOURNAL_CONTENT_QUERY = (
    "((microglia AND (morphology OR spatial transcriptomics OR gene expression)) OR "
    "(cell morphology AND (transcriptomics OR representation learning)) OR "
    "(multimodal AND (single cell OR spatial omics OR microscopy)))"
)

MAX_RECENT  = 40
MAX_CLASSIC = 15
HISTORY_FILE = "history.md"
CLASSIC_YEAR_CUTOFF = datetime.now().year - 3
RECENT_YEAR_CUTOFF  = datetime.now().year - 1

# Sources that are named journals (not preprint servers)
JOURNAL_SOURCES = {"PubMed", "Semantic Scholar", "arXiv"}  # arXiv excluded below in sort


# ── History ──────────────────────────────────────────────────────────────────

def load_history() -> tuple[set[str], list[str]]:
    """Returns (all_title_keys, last_7_days_titles)."""
    if not os.path.exists(HISTORY_FILE):
        return set(), []
    keys = set()
    recent_titles = []
    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.+?)\s*\|", line)
            if m:
                date_str, title = m.group(1), m.group(2).strip()
                keys.add(title.lower()[:70])
                if date_str >= cutoff:
                    recent_titles.append(title)
    return keys, recent_titles


def save_history(papers: list[dict]):
    today = datetime.now().strftime("%Y-%m-%d")
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        for p in papers:
            tag = p.get("must_read_tag", "⭐" if p.get("must_read") else "")
            title = p["title"].replace("|", "\\|")
            f.write(f"| {today} | {title} | {p['url']} | {p['source']} | {tag} |\n")


# ── Domain Tagger ─────────────────────────────────────────────────────────────

def tag_domain(p: dict) -> str:
    text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
    if any(k in text for k in ["benchmark", "held-out", "held out", "cross-dataset",
                                "external validation", "negative control"]):
        return "Benchmark"
    if any(k in text for k in ["microglia", "microglial", "ramification", "ameboid",
                                "amoeboid", "process complexity"]):
        return "MicrogliaMorphology"
    if any(k in text for k in ["contrastive", "multimodal", "multi-modal", "optimal transport",
                                "adversarial autoencoder", "image encoder", "graph neural network",
                                "patch-seq", "morphology transcriptomics"]):
        return "MultimodalAI"
    if any(k in text for k in ["spatial transcriptomics", "spatial omics", "visium",
                                "gene expression", "single cell", "scrna", "rna-seq"]):
        return "SpatialOmics"
    return "other"


# ── Fetchers ─────────────────────────────────────────────────────────────────

def fetch_arxiv(query: str, n: int = 15) -> list[dict]:
    search = f"({query.replace(' ', '+')}) AND ({ARXIV_CATS})"
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query={quote(search)}"
        f"&start=0&max_results={n}&sortBy=submittedDate&sortOrder=descending"
    )
    feed = feedparser.parse(url)
    results = []
    for e in feed.entries:
        year = None
        if hasattr(e, "published"):
            try:
                year = int(e.published[:4])
            except Exception:
                pass
        results.append({
            "title": e.title.replace("\n", " ").strip(),
            "abstract": e.summary.replace("\n", " ")[:600].strip(),
            "authors": ", ".join(a.name for a in e.authors[:3]),
            "url": e.link,
            "source": "arXiv",
            "year": year,
        })
    return results


def fetch_semantic_scholar(query: str, n: int = 10, min_year: int = None, max_year: int = None,
                            sort: str = "relevance") -> list[dict]:
    params = {
        "query": query,
        "fields": "title,abstract,authors,year,url,citationCount",
        "limit": n,
    }
    if sort == "citations":
        params["sort"] = "citationCount"
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params, timeout=12,
        )
        r.raise_for_status()
        papers = []
        for p in r.json().get("data", []):
            if not p.get("abstract"):
                continue
            year = p.get("year")
            if min_year and year and year < min_year:
                continue
            if max_year and year and year > max_year:
                continue
            papers.append({
                "title": p["title"],
                "abstract": p["abstract"][:600],
                "authors": ", ".join(a["name"] for a in p.get("authors", [])[:3]),
                "url": p.get("url", ""),
                "source": "Semantic Scholar",
                "year": year,
                "citations": p.get("citationCount", 0),
            })
        return papers
    except Exception as e:
        print(f"[SS] {e}")
        return []


def _parse_pubmed_xml(ids: list[str], xml_text: str, default_source: str = "PubMed") -> list[dict]:
    """Parse PubMed efetch XML into paper dicts. Extracts journal name when present."""
    papers = []
    for uid in ids:
        title_m = re.search(
            rf"<PubmedArticle>.*?<PMID[^>]*>{uid}</PMID>.*?<ArticleTitle>(.*?)</ArticleTitle>",
            xml_text, re.DOTALL
        )
        if not title_m:
            continue
        abstract_m = re.search(
            rf"<PubmedArticle>.*?<PMID[^>]*>{uid}</PMID>.*?<AbstractText[^>]*>(.*?)</AbstractText>",
            xml_text, re.DOTALL
        )
        author_m = re.search(
            rf"<PubmedArticle>.*?<PMID[^>]*>{uid}</PMID>.*?<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>",
            xml_text, re.DOTALL
        )
        year_m = re.search(
            rf"<PubmedArticle>.*?<PMID[^>]*>{uid}</PMID>.*?<PubDate>.*?<Year>(\d{{4}})</Year>",
            xml_text, re.DOTALL
        )
        journal_m = re.search(
            rf"<PubmedArticle>.*?<PMID[^>]*>{uid}</PMID>.*?<Journal>.*?<Title>(.*?)</Title>",
            xml_text, re.DOTALL
        )
        title    = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()
        abstract = re.sub(r"<[^>]+>", "", abstract_m.group(1)).strip()[:600] if abstract_m else ""
        author   = f"{author_m.group(1)} {author_m.group(2)}" if author_m else ""
        year     = int(year_m.group(1)) if year_m else None
        source   = journal_m.group(1).strip() if journal_m else default_source
        papers.append({
            "title": title,
            "abstract": abstract,
            "authors": author,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            "source": source,
            "year": year,
        })
    return papers


def fetch_pubmed(query: str, n: int = 8) -> list[dict]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        ids = requests.get(
            f"{base}/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": n,
                    "sort": "pub+date", "retmode": "json", "reldate": 90},
            timeout=12,
        ).json()["esearchresult"]["idlist"]
        if not ids:
            return []
        fetch_r = requests.get(
            f"{base}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"},
            timeout=15,
        )
        return _parse_pubmed_xml(ids, fetch_r.text, default_source="PubMed")
    except Exception as e:
        print(f"[PubMed] {e}")
        return []


def fetch_pubmed_journals(n: int = 12) -> list[dict]:
    """Fetch recent papers from top journals using a journal whitelist filter."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    term = f"({JOURNAL_CONTENT_QUERY}) AND ({JOURNAL_FILTER})"
    try:
        ids = requests.get(
            f"{base}/esearch.fcgi",
            params={"db": "pubmed", "term": term, "retmax": n,
                    "sort": "pub+date", "retmode": "json", "reldate": 180},
            timeout=12,
        ).json()["esearchresult"]["idlist"]
        if not ids:
            print("[Journals] No results")
            return []
        fetch_r = requests.get(
            f"{base}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"},
            timeout=15,
        )
        papers = _parse_pubmed_xml(ids, fetch_r.text, default_source="PubMed")
        print(f"[Journals] {len(papers)} papers from: {set(p['source'] for p in papers)}")
        return papers
    except Exception as e:
        print(f"[Journals] {e}")
        return []


def deduplicate(papers: list[dict], history_keys: set[str]) -> list[dict]:
    seen, unique = set(), []
    for p in papers:
        key = p["title"].lower()[:70]
        if key not in seen and key not in history_keys and p["title"]:
            seen.add(key)
            unique.append(p)
    return unique


def balance_pool(papers: list[dict], cap: int) -> list[dict]:
    """Interleave by domain; within each domain, journal papers first."""
    domains = ["MicrogliaMorphology", "SpatialOmics", "MultimodalAI", "Benchmark", "other"]
    buckets: dict[str, list] = {domain: [] for domain in domains}
    for p in papers:
        buckets[p.get("domain", "other")].append(p)

    def journal_first(lst):
        return (
            sorted([p for p in lst if p["source"] not in ("arXiv",)], key=lambda p: p.get("year") or 0, reverse=True) +
            [p for p in lst if p["source"] == "arXiv"]
        )

    balanced = []
    per_domain = max(8, cap // len(domains))
    for domain in domains:
        balanced.extend(journal_first(buckets[domain])[:per_domain])
    return balanced[:cap]


# ── DeepSeek Selection ───────────────────────────────────────────────────────

RESEARCHER_BIO = """PhD student in Biostatistics, University of Pittsburgh.
Current research:
1. Biological question: when microglial morphology and molecular state are coupled versus decoupled, especially in aging and Alzheimer's disease.
2. Data: segmented microglia from multiplex immunofluorescence images, Visium with protein co-detection, spatial transcriptomics, and potentially unpaired morphology/transcriptomic datasets.
3. AI methods: classical morphology features, CAJAL, image encoders, skeleton GNNs; CCA, optimal transport, contrastive learning, and adversarial representation alignment such as GeoAdvAE.
4. Benchmark design: donor-held-out, region-held-out, and dataset-held-out evaluation, plus permutation, batch-only prediction, and staining-quality negative controls.
The dissertation should emphasize reusable AI-for-Science methodology while answering a concrete biological question."""


def select_papers(recent: list[dict], classics: list[dict], recent_history: list[str]) -> list[dict]:
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    def fmt(papers, label):
        return "\n\n".join(
            f"[{label}{i+1}] {p['title']} ({p.get('year','?')})\n"
            f"Authors: {p['authors']} | Source: {p['source']} | Domain: {p.get('domain','?')}"
            + (f" | Citations: {p['citations']}" if p.get('citations') else "") + "\n"
            f"Abstract: {p['abstract']}\n"
            f"URL: {p['url']}"
            for i, p in enumerate(papers)
        )

    history_note = ""
    if recent_history:
        titles = "\n".join(f"- {t}" for t in recent_history[-14:])
        history_note = f"\nPapers recommended in the last 7 days (avoid thematic repetition):\n{titles}\n"

    diversity_rule = """
Relevance rule: Every selected paper must directly help the dissertation described above. Exclude generic clinical-prediction papers that do not inform this project.
Coverage rule: The three papers should complement each other and, when strong candidates exist, cover at least two different domains from this list:
  - MicrogliaMorphology (microglial biology, morphology, activation, aging, Alzheimer's disease)
  - SpatialOmics (spatial transcriptomics, Visium, single-cell molecular states)
  - MultimodalAI (morphology–expression alignment, Patch-seq, CCA, OT, contrastive/adversarial learning, image or graph encoders)
  - Benchmark (cross-donor/region/dataset generalization, controls, reproducibility)
Journal preference rule: When quality is comparable, STRONGLY prefer papers from named journals \
(Nature, Nature Methods, Nature Biotechnology, Cell, Genome Biology, Science, NEJM, Lancet, JAMA, \
Bioinformatics, etc.) over arXiv preprints for the RECENT slots.
Reading-brief rule: Follow the first-pass goals in Keshav's three-pass reading method. Help the reader quickly identify the problem, approach, evidence, contribution, and whether the paper deserves a deeper read. Base every statement only on the supplied metadata and abstract; explicitly say when a detail is unavailable rather than inventing it.
For every selected paper, write substantial but scannable Chinese:
  - one_liner: one sentence stating the paper's question, approach, and headline result.
  - what: 3–5 sentences covering the research question, data/experimental setting, method, and main result.
  - innovation: 2–4 sentences explaining the concrete novelty relative to prior practice; distinguish a new biological finding from a new computational method.
  - learn: 2–4 sentences tailored to this dissertation, naming concepts/methods/figures to study and one question to keep in mind while reading.
"""

    has_classics = len(classics) > 0
    if has_classics:
        task = f"""Select exactly 3 papers total — 2 from the RECENT pool and 1 from the CLASSIC pool.
{diversity_rule}
Marking rules:
- Assign "⭐ 今日精读" to the single paper that should be read first.
- Assign must_read_tag "" to the other paper.

Return ONLY valid JSON, no markdown fences:
{{
  "papers": [
    {{"pool": "recent", "index": <1-based in RECENT>, "domain": "<domain>", "must_read_tag": "⭐ 今日精读" or "", "one_liner": "<中文>", "what": "<中文>", "innovation": "<中文>", "learn": "<中文>"}},
    {{"pool": "recent", "index": <different 1-based index>, "domain": "<domain>", "must_read_tag": "⭐ 今日精读" or "", "one_liner": "<中文>", "what": "<中文>", "innovation": "<中文>", "learn": "<中文>"}},
    {{"pool": "classic", "index": <1-based in CLASSIC>, "domain": "<domain>", "must_read_tag": "⭐ 今日精读" or "", "one_liner": "<中文>", "what": "<中文>", "innovation": "<中文>", "learn": "<中文>"}}
  ]
}}

RECENT papers ({len(recent)} candidates):
{fmt(recent, 'R')}

CLASSIC papers ({len(classics)} candidates):
{fmt(classics, 'C')}"""
    else:
        task = f"""Select exactly 3 papers from the RECENT pool.
{diversity_rule}
Marking rules:
- Assign "⭐ 今日精读" to the single paper that should be read first.
- Assign must_read_tag "" to the other paper.

Return ONLY valid JSON, no markdown fences:
{{
  "papers": [
    {{"pool": "recent", "index": <1-based>, "domain": "<domain>", "must_read_tag": "⭐ 今日精读" or "", "one_liner": "<中文>", "what": "<中文>", "innovation": "<中文>", "learn": "<中文>"}},
    {{"pool": "recent", "index": <different 1-based index>, "domain": "<domain>", "must_read_tag": "⭐ 今日精读" or "", "one_liner": "<中文>", "what": "<中文>", "innovation": "<中文>", "learn": "<中文>"}},
    {{"pool": "recent", "index": <another different 1-based index>, "domain": "<domain>", "must_read_tag": "⭐ 今日精读" or "", "one_liner": "<中文>", "what": "<中文>", "innovation": "<中文>", "learn": "<中文>"}}
  ]
}}

RECENT papers ({len(recent)} candidates):
{fmt(recent, 'R')}"""

    prompt = f"""You are a research paper recommendation assistant for this researcher:
{RESEARCHER_BIO}
{history_note}
{task}"""

    models = [os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")]
    messages = [
        {"role": "system", "content": "You are a research paper recommendation assistant. Always respond with valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    result = None
    for model in models:
        for attempt in range(2):
            resp = client.chat.completions.create(
                model=model, max_tokens=3500, messages=messages,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            text = (resp.choices[0].message.content or "").strip()
            text = text.lstrip("```json").lstrip("```").rstrip("```").strip()
            print(f"[{model} attempt {attempt+1}] {text[:300]}")
            if not text:
                print(f"{model} returned empty, trying next model...")
                break
            try:
                result = json.loads(text)
                print(f"✓ Parsed with {model}")
                break
            except json.JSONDecodeError:
                if attempt == 0:
                    print("JSON parse failed, retrying...")
                    time.sleep(2)
        if result:
            break

    if not result:
        raise RuntimeError("DeepSeek failed to return valid JSON")

    selected = []
    for item in result["papers"][:3]:
        pool = item.get("pool", "recent")
        idx = item["index"] - 1
        source_pool = recent if pool == "recent" else classics
        if idx < 0 or idx >= len(source_pool):
            continue
        p = source_pool[idx].copy()
        p["must_read_tag"] = item.get("must_read_tag", "")
        p["must_read"] = bool(p["must_read_tag"])
        p["one_liner"] = item.get("one_liner", "")
        p["what"] = item.get("what", item.get("why", ""))
        p["innovation"] = item.get("innovation", "摘要未提供足够信息判断具体创新。")
        p["learn"] = item.get("learn", item.get("why", ""))
        selected.append(p)
    return selected


# ── Email ─────────────────────────────────────────────────────────────────────

REPO_URL = "https://github.com/liangyou03/daily-paper"

def build_html(papers: list[dict]) -> str:
    today = datetime.now().strftime("%Y年%m月%d日")
    cards = ""
    domain_labels = {
        "MicrogliaMorphology": "MICROGLIA · MORPHOLOGY",
        "SpatialOmics": "SPATIAL OMICS",
        "MultimodalAI": "MULTIMODAL AI",
        "Benchmark": "BENCHMARK",
        "other": "RELATED WORK",
    }
    for number, p in enumerate(papers, start=1):
        tag = p.get("must_read_tag", "")
        if tag:
            badge = f'<span style="background:#cc785c;color:#fffaf4;padding:4px 10px;border-radius:999px;font-size:11px;font-weight:700;">{html.escape(tag)}</span>'
            border = "border:1px solid #d9b4a6;"
        else:
            badge = ""
            border = "border:1px solid #ded8cf;"

        year_str = f" · {p['year']}" if p.get("year") else ""
        title = html.escape(p["title"])
        authors = html.escape(p.get("authors", ""))
        source = html.escape(f"{p['source']}{year_str}")
        url = html.escape(p["url"], quote=True)
        domain = html.escape(domain_labels.get(p.get("domain", "other"), p.get("domain", "RELATED WORK")))
        one_liner = html.escape(p.get("one_liner", p.get("why", "")))
        what = html.escape(p.get("what", p.get("why", "摘要未提供。")))
        innovation = html.escape(p.get("innovation", "摘要未提供足够信息判断具体创新。"))
        learn = html.escape(p.get("learn", p.get("why", "建议先阅读摘要、图表和结论。")))
        cards += f"""
        <div style="background:#fffdfa;{border}border-radius:14px;padding:25px;margin-bottom:18px;box-shadow:0 5px 18px rgba(58,48,42,.055);">
          <div style="margin-bottom:12px;">
            <span style="display:inline-block;background:#cc785c;color:#fffaf4;width:25px;height:25px;line-height:25px;text-align:center;border-radius:50%;font-size:12px;font-weight:700;margin-right:7px;">{number}</span>
            <span style="color:#9a5b46;font-size:10px;font-weight:800;letter-spacing:1px;">{domain}</span>
            <span style="float:right;">{badge}</span>
          </div>
          <h2 style="margin:0 0 7px;font-family:Georgia,'Times New Roman',serif;font-size:18px;font-weight:500;line-height:1.45;">
            <a href="{url}" style="color:#2d2a26;text-decoration:none;">{title}</a>
          </h2>
          <p style="margin:0 0 16px;color:#756f68;font-size:11px;line-height:1.55;">{authors}<br>{source}</p>
          <div style="background:#f3e6df;border-left:4px solid #cc785c;padding:13px 15px;border-radius:0 8px 8px 0;margin-bottom:18px;color:#3d302b;font-size:13px;font-weight:600;line-height:1.75;">
            <span style="display:block;color:#9a5b46;font-size:10px;font-weight:800;letter-spacing:.8px;margin-bottom:3px;">一句话看懂</span>{one_liner}
          </div>
          <div style="margin-bottom:14px;">
            <div style="color:#9a5b46;font-size:12px;font-weight:800;margin-bottom:5px;">01 · 论文做了什么</div>
            <div style="color:#3d3a36;font-size:13px;line-height:1.8;">{what}</div>
          </div>
          <div style="margin-bottom:14px;">
            <div style="color:#806b5c;font-size:12px;font-weight:800;margin-bottom:5px;">02 · 创新在哪里</div>
            <div style="color:#3d3a36;font-size:13px;line-height:1.8;">{innovation}</div>
          </div>
          <div style="background:#f2eee7;border-radius:9px;padding:13px 15px;">
            <div style="color:#9a5b46;font-size:12px;font-weight:800;margin-bottom:5px;">03 · 你应该学什么</div>
            <div style="color:#3d3a36;font-size:13px;line-height:1.8;">{learn}</div>
          </div>
        </div>"""

    must_count = sum(1 for p in papers if p.get("must_read"))
    return f"""<html><body style="margin:0;padding:24px 12px;background:#f7f4ed;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif;">
<div style="max-width:650px;margin:0 auto;">
  <div style="background:#2d2a26;color:#fffaf4;padding:29px 27px;border-radius:16px;margin-bottom:18px;box-shadow:0 8px 24px rgba(58,48,42,.16);">
    <div style="color:#e7a58e;font-size:10px;font-weight:800;letter-spacing:1.4px;margin-bottom:9px;">DISSERTATION READING BRIEF</div>
    <h1 style="margin:0 0 9px;font-family:Georgia,'Times New Roman',serif;font-size:24px;font-weight:500;line-height:1.3;">Microglia × Multimodal AI</h1>
    <p style="margin:0;color:#d8d1c8;font-size:12px;line-height:1.65;">{today} · {len(papers)} 篇精选 · {must_count} 篇今日精读<br>Morphology · Molecular State · Spatial Omics · Benchmark</p>
  </div>
  {cards}
  <p style="text-align:center;color:#8b8279;font-size:11px;line-height:1.7;margin:22px 0;">
    按 Keshav three-pass method 设计的第一遍阅读摘要 · Powered by DeepSeek<br>
    <a href="{REPO_URL}/blob/main/history.md" style="color:#b5654b;text-decoration:none;">查看往期推荐 →</a>
  </p>
</div>
</body></html>"""


def send_email(papers: list[dict]):
    gmail_user = os.environ["GMAIL_USER"]
    gmail_pass = os.environ["GMAIL_APP_PASSWORD"]
    to_email   = os.environ.get("TO_EMAIL", gmail_user)

    must_count = sum(1 for p in papers if p.get("must_read"))
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📚 论文推荐 {datetime.now().strftime('%m/%d')} · {len(papers)} 篇 · {must_count} 篇精读"
    msg["From"]    = gmail_user
    msg["To"]      = to_email
    msg.attach(MIMEText(build_html(papers), "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(gmail_user, gmail_pass)
        s.sendmail(gmail_user, to_email, msg.as_string())
    print(f"✅ Sent to {to_email}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    history_keys, recent_history = load_history()
    print(f"History: {len(history_keys)} papers seen, {len(recent_history)} in last 7 days")

    # ── Recent pool ──
    recent_pool = []

    print("Fetching arXiv (recent)...")
    for q in ARXIV_QUERIES:
        recent_pool.extend(fetch_arxiv(q))
        time.sleep(1)

    print("Fetching PubMed (topic queries)...")
    for q in PUBMED_QUERIES:
        recent_pool.extend(fetch_pubmed(q))
        time.sleep(0.5)

    print("Fetching PubMed top journals...")
    recent_pool.extend(fetch_pubmed_journals(n=12))
    time.sleep(0.5)

    # Tag domains before dedup/balancing
    for p in recent_pool:
        p["domain"] = tag_domain(p)

    recent_pool = deduplicate(recent_pool, history_keys)
    print(f"Recent candidates after dedup: {len(recent_pool)}")
    recent_pool = balance_pool(recent_pool, MAX_RECENT)
    domains = ["MicrogliaMorphology", "SpatialOmics", "MultimodalAI", "Benchmark", "other"]
    print(f"Recent pool after balancing: {len(recent_pool)} | domains: { {d: sum(1 for p in recent_pool if p.get('domain')==d) for d in domains} }")

    # ── Classic pool ──
    classic_pool = []
    print("Fetching Semantic Scholar (classics)...")
    for q in SS_CLASSIC_QUERIES:
        classic_pool.extend(
            fetch_semantic_scholar(q, n=8, max_year=CLASSIC_YEAR_CUTOFF, sort="citations")
        )
        time.sleep(3)

    for p in classic_pool:
        p["domain"] = tag_domain(p)

    classic_pool = deduplicate(classic_pool, history_keys)
    classic_pool.sort(key=lambda p: p.get("citations", 0), reverse=True)
    classic_pool = classic_pool[:MAX_CLASSIC]
    print(f"Classic candidates after dedup: {len(classic_pool)}")

    print("Asking DeepSeek to select...")
    selected = select_papers(recent_pool, classic_pool, recent_history)

    print("Sending email...")
    send_email(selected)

    print("Saving history...")
    save_history(selected)


if __name__ == "__main__":
    main()
