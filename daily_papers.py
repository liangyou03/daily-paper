#!/usr/bin/env python3
"""Daily paper digest: UQ · AI4Health · AI4Omics"""

import os, json, time, smtplib, re, requests, feedparser
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import quote
from openai import OpenAI

# ── Search Config ────────────────────────────────────────────────────────────

ARXIV_QUERIES = [
    "uncertainty quantification machine learning",
    "conformal prediction",
    "AI healthcare clinical prediction",
    "single cell RNA sequencing deep learning",
    "spatial transcriptomics",
    "omics foundation model",
]
ARXIV_CATS = (
    "cat:cs.LG OR cat:stat.ML OR cat:stat.ME "
    "OR cat:q-bio.QM OR cat:q-bio.GN OR cat:cs.AI"
)

SS_QUERIES = [
    "uncertainty quantification deep learning",
    "machine learning electronic health records",
    "single cell omics transformer",
]

# Classic queries: foundational topics likely to yield high-citation older papers
SS_CLASSIC_QUERIES = [
    "conformal prediction coverage guarantee",
    "deep learning electronic health records mortality prediction",
    "single cell RNA sequencing dimensionality reduction",
]

PUBMED_QUERIES = [
    "deep learning single cell RNA sequencing",
    "uncertainty quantification clinical prediction model",
    "spatial transcriptomics machine learning",
]

MAX_RECENT = 40    # recent pool cap (arXiv + SS + PubMed)
MAX_CLASSIC = 15   # classics pool cap
HISTORY_FILE = "history.md"
CLASSIC_YEAR_CUTOFF = datetime.now().year - 3  # papers from ≤ this year are "classic"
RECENT_YEAR_CUTOFF  = datetime.now().year - 1  # papers from ≥ this year are "recent"


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
        # Parse year from published date (e.g. "2024-03-15T...")
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
            params=params,
            timeout=12,
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
        # Use efetch for real abstracts
        fetch_r = requests.get(
            f"{base}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"},
            timeout=15,
        )
        papers = []
        # Parse XML minimally to get title + abstract
        for uid in ids:
            title_m = re.search(
                rf"<PubmedArticle>.*?<PMID[^>]*>{uid}</PMID>.*?<ArticleTitle>(.*?)</ArticleTitle>",
                fetch_r.text, re.DOTALL
            )
            abstract_m = re.search(
                rf"<PubmedArticle>.*?<PMID[^>]*>{uid}</PMID>.*?<AbstractText[^>]*>(.*?)</AbstractText>",
                fetch_r.text, re.DOTALL
            )
            author_m = re.search(
                rf"<PubmedArticle>.*?<PMID[^>]*>{uid}</PMID>.*?<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>",
                fetch_r.text, re.DOTALL
            )
            year_m = re.search(
                rf"<PubmedArticle>.*?<PMID[^>]*>{uid}</PMID>.*?<PubDate>.*?<Year>(\d{{4}})</Year>",
                fetch_r.text, re.DOTALL
            )
            if not title_m:
                continue
            title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()
            abstract = re.sub(r"<[^>]+>", "", abstract_m.group(1)).strip()[:600] if abstract_m else ""
            author = f"{author_m.group(1)} {author_m.group(2)}" if author_m else ""
            year = int(year_m.group(1)) if year_m else None
            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": author,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                "source": "PubMed",
                "year": year,
            })
        return papers
    except Exception as e:
        print(f"[PubMed] {e}")
        return []


def deduplicate(papers: list[dict], history_keys: set[str]) -> list[dict]:
    seen, unique = set(), []
    for p in papers:
        key = p["title"].lower()[:70]
        if key not in seen and key not in history_keys and p["title"]:
            seen.add(key)
            unique.append(p)
    return unique


# ── GLM Selection ────────────────────────────────────────────────────────────

RESEARCHER_BIO = """PhD student in Biostatistics, University of Pittsburgh.
Current research:
1. Uncertainty Quantification / Conformal Prediction — especially for simplex-valued predictions and clinical ICU data with missingness (Mondrian CP, coverage theory)
2. ML for Healthcare — ICU outcome prediction, domain shift, GOSSIS/MIMIC datasets
3. Computational Biology — spatial transcriptomics, single-cell omics, cell segmentation, IF imaging
Target venues: NeurIPS, ICML, ICLR (Datasets & Benchmarks track included)."""


def select_papers(recent: list[dict], classics: list[dict], recent_history: list[str]) -> list[dict]:
    client = OpenAI(
        api_key=os.environ["GLM_API_KEY"],
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    def fmt(papers, label):
        return "\n\n".join(
            f"[{label}{i+1}] {p['title']} ({p.get('year','?')})\n"
            f"Authors: {p['authors']} | Source: {p['source']}"
            + (f" | Citations: {p['citations']}" if p.get('citations') else "") + "\n"
            f"Abstract: {p['abstract']}\n"
            f"URL: {p['url']}"
            for i, p in enumerate(papers)
        )

    history_note = ""
    if recent_history:
        titles = "\n".join(f"- {t}" for t in recent_history[-14:])
        history_note = f"\nPapers recommended in the last 7 days (avoid thematic repetition):\n{titles}\n"

    prompt = f"""You are a research paper recommendation assistant for this researcher:
{RESEARCHER_BIO}
{history_note}
Select exactly 5 papers total — 3 from the RECENT pool and 2 from the CLASSIC pool.

Marking rules:
- must_read_tag "⭐ 近期精读" → the single most impactful RECENT paper (≤1 year old)
- must_read_tag "⭐ 经典精读" → the single most foundational CLASSIC paper (≥3 years old, high citation)
- All other papers: must_read_tag ""

Return ONLY valid JSON, no markdown fences:
{{
  "papers": [
    {{"pool": "recent", "index": <1-based in RECENT>, "must_read_tag": "⭐ 近期精读" or "", "why": "<2-3句中文推荐理由>"}},
    ...3 recent entries...,
    {{"pool": "classic", "index": <1-based in CLASSIC>, "must_read_tag": "⭐ 经典精读" or "", "why": "<2-3句中文推荐理由>"}},
    ...2 classic entries...
  ]
}}

RECENT papers ({len(recent)} candidates):
{fmt(recent, 'R')}

CLASSIC papers ({len(classics)} candidates):
{fmt(classics, 'C')}"""

    models = ["glm-5", "glm-4-plus"]
    messages = [
        {"role": "system", "content": "You are a research paper recommendation assistant. Always respond with valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    result = None
    for model in models:
        for attempt in range(2):
            resp = client.chat.completions.create(
                model=model, max_tokens=1500, messages=messages,
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
        raise RuntimeError("All GLM models failed to return valid JSON")

    selected = []
    for item in result["papers"]:
        pool = item.get("pool", "recent")
        idx = item["index"] - 1
        source = recent if pool == "recent" else classics
        if idx < 0 or idx >= len(source):
            continue
        p = source[idx].copy()
        p["must_read_tag"] = item.get("must_read_tag", "")
        p["must_read"] = bool(p["must_read_tag"])
        p["why"] = item.get("why", "")
        selected.append(p)
    return selected


# ── Email ─────────────────────────────────────────────────────────────────────

REPO_URL = "https://github.com/liangyou03/daily-paper"

def build_html(papers: list[dict]) -> str:
    today = datetime.now().strftime("%Y年%m月%d日")
    cards = ""
    for p in papers:
        tag = p.get("must_read_tag", "")
        if tag:
            badge = f'<span style="background:#c62828;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:.5px;">{tag}</span> '
            border = "border-left:4px solid #c62828;"
            bg = "background:#fff8f8;"
        else:
            badge = ""
            border = "border-left:4px solid #e0e0e0;"
            bg = "background:#fff;"

        year_str = f" · {p['year']}" if p.get("year") else ""
        cards += f"""
        <div style="{bg}{border}border-radius:6px;padding:18px 20px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.08);">
          <div style="margin-bottom:6px;">{badge}<span style="color:#888;font-size:11px;">{p['source']}{year_str}</span></div>
          <h3 style="margin:0 0 5px;font-size:14px;line-height:1.5;">
            <a href="{p['url']}" style="color:#1565c0;text-decoration:none;">{p['title']}</a>
          </h3>
          <p style="margin:0 0 10px;color:#777;font-size:11px;">{p['authors']}</p>
          <div style="background:#f5f5f5;padding:10px 12px;border-radius:4px;font-size:12px;color:#444;line-height:1.6;">
            💡 {p['why']}
          </div>
        </div>"""

    must_count = sum(1 for p in papers if p.get("must_read"))
    return f"""<html><body style="margin:0;padding:20px;background:#f0f2f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
<div style="max-width:580px;margin:0 auto;">
  <div style="background:#1565c0;color:#fff;padding:18px 22px;border-radius:8px 8px 0 0;">
    <h2 style="margin:0;font-size:17px;">📚 每日论文推荐 · {today}</h2>
    <p style="margin:4px 0 0;font-size:12px;opacity:.8;">UQ · AI4Health · AI4Omics | {len(papers)} 篇推荐 · {must_count} 篇精读 | Powered by GLM</p>
  </div>
  <div style="padding:14px 0;">{cards}</div>
  <p style="text-align:center;color:#aaa;font-size:11px;margin-top:4px;">
    每日自动推送 · 为你的研究方向定制 ·
    <a href="{REPO_URL}/blob/main/history.md" style="color:#aaa;">往期推荐</a>
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

    print("Fetching Semantic Scholar (recent)...")
    for q in SS_QUERIES:
        recent_pool.extend(fetch_semantic_scholar(q, min_year=RECENT_YEAR_CUTOFF))
        time.sleep(2)

    print("Fetching PubMed (recent)...")
    for q in PUBMED_QUERIES:
        recent_pool.extend(fetch_pubmed(q))
        time.sleep(0.5)

    recent_pool = deduplicate(recent_pool, history_keys)
    print(f"Recent candidates after dedup: {len(recent_pool)}")
    # Prioritize: keep up to MAX_RECENT, prefer arXiv first (most recent)
    arxiv_papers  = [p for p in recent_pool if p["source"] == "arXiv"]
    other_papers  = [p for p in recent_pool if p["source"] != "arXiv"]
    recent_pool   = (arxiv_papers + other_papers)[:MAX_RECENT]

    # ── Classic pool ──
    classic_pool = []
    print("Fetching Semantic Scholar (classics)...")
    for q in SS_CLASSIC_QUERIES:
        classic_pool.extend(
            fetch_semantic_scholar(q, n=8, max_year=CLASSIC_YEAR_CUTOFF, sort="citations")
        )
        time.sleep(2)

    classic_pool = deduplicate(classic_pool, history_keys)
    # Sort classics by citation count descending
    classic_pool.sort(key=lambda p: p.get("citations", 0), reverse=True)
    classic_pool = classic_pool[:MAX_CLASSIC]
    print(f"Classic candidates after dedup: {len(classic_pool)}")

    print("Asking GLM to select...")
    selected = select_papers(recent_pool, classic_pool, recent_history)

    print("Sending email...")
    send_email(selected)

    print("Saving history...")
    save_history(selected)


if __name__ == "__main__":
    main()
