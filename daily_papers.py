#!/usr/bin/env python3
"""Daily paper digest: UQ · AI4Health · AI4Omics"""

import os, json, time, random, smtplib, requests, feedparser
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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

PUBMED_QUERIES = [
    "deep learning single cell RNA sequencing",
    "uncertainty quantification clinical prediction model",
    "spatial transcriptomics machine learning",
]

MAX_CANDIDATES = 35  # cap before Claude selection


# ── Fetchers ─────────────────────────────────────────────────────────────────

def fetch_arxiv(query: str, n: int = 15) -> list[dict]:
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=({query.replace(' ', '+')}) AND ({ARXIV_CATS})"
        f"&start=0&max_results={n}&sortBy=submittedDate&sortOrder=descending"
    )
    feed = feedparser.parse(url)
    return [
        {
            "title": e.title.replace("\n", " ").strip(),
            "abstract": e.summary.replace("\n", " ")[:600].strip(),
            "authors": ", ".join(a.name for a in e.authors[:3]),
            "url": e.link,
            "source": "arXiv",
        }
        for e in feed.entries
    ]


def fetch_semantic_scholar(query: str, n: int = 10) -> list[dict]:
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "fields": "title,abstract,authors,year,url", "limit": n},
            timeout=12,
        )
        r.raise_for_status()
        papers = []
        for p in r.json().get("data", []):
            if not p.get("abstract"):
                continue
            papers.append({
                "title": p["title"],
                "abstract": p["abstract"][:600],
                "authors": ", ".join(a["name"] for a in p.get("authors", [])[:3]),
                "url": p.get("url", ""),
                "source": "Semantic Scholar",
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
                    "sort": "pub+date", "retmode": "json", "reldate": 60},
            timeout=12,
        ).json()["esearchresult"]["idlist"]
        if not ids:
            return []
        result = requests.get(
            f"{base}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=12,
        ).json()["result"]
        return [
            {
                "title": result[uid].get("title", ""),
                "abstract": result[uid].get("source", ""),
                "authors": ", ".join(
                    a["name"] for a in result[uid].get("authors", [])[:3]
                ),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                "source": "PubMed",
            }
            for uid in ids
            if uid in result and result[uid].get("title")
        ]
    except Exception as e:
        print(f"[PubMed] {e}")
        return []


def deduplicate(papers: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for p in papers:
        key = p["title"].lower()[:70]
        if key not in seen and p["title"]:
            seen.add(key)
            unique.append(p)
    return unique


# ── Claude Selection ──────────────────────────────────────────────────────────

RESEARCHER_BIO = """PhD student in Biostatistics, University of Pittsburgh.
Current research:
1. Uncertainty Quantification / Conformal Prediction — especially for simplex-valued predictions and clinical ICU data with missingness (Mondrian CP, coverage theory)
2. ML for Healthcare — ICU outcome prediction, domain shift, GOSSIS/MIMIC datasets
3. Computational Biology — spatial transcriptomics, single-cell omics, cell segmentation, IF imaging
Target venues: NeurIPS, ICML, ICLR (Datasets & Benchmarks track included)."""


def select_papers(candidates: list[dict]) -> list[dict]:
    client = OpenAI(
        api_key=os.environ["GLM_API_KEY"],
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    listing = "\n\n".join(
        f"[{i+1}] {p['title']}\n"
        f"Authors: {p['authors']} | Source: {p['source']}\n"
        f"Abstract: {p['abstract']}\n"
        f"URL: {p['url']}"
        for i, p in enumerate(candidates)
    )

    prompt = f"""You are a research assistant for this researcher:
{RESEARCHER_BIO}

From the {len(candidates)} papers below, pick exactly 3 that are most relevant and impactful.
Mark exactly 1 as must_read (highest relevance or breakthrough result).

Return ONLY valid JSON, no markdown fences:
{{
  "papers": [
    {{"index": <1-based>, "must_read": true/false, "why": "<2-3句中文推荐理由，说明与该研究者当前工作的关联>"}}
  ]
}}

Papers:
{listing}"""

    resp = client.chat.completions.create(
        model="glm-4-plus",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip().lstrip("```json").lstrip("```").rstrip("```")
    result = json.loads(text)

    selected = []
    for item in result["papers"]:
        p = candidates[item["index"] - 1].copy()
        p["must_read"] = item["must_read"]
        p["why"] = item["why"]
        selected.append(p)
    return selected


# ── Email ─────────────────────────────────────────────────────────────────────

def build_html(papers: list[dict]) -> str:
    today = datetime.now().strftime("%Y年%m月%d日")
    cards = ""
    for p in papers:
        if p["must_read"]:
            badge = '<span style="background:#c62828;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:.5px;">⭐ 强烈推荐精读</span> '
            border = "border-left:4px solid #c62828;"
            bg = "background:#fff8f8;"
        else:
            badge = ""
            border = "border-left:4px solid #e0e0e0;"
            bg = "background:#fff;"

        cards += f"""
        <div style="{bg}{border}border-radius:6px;padding:18px 20px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.08);">
          <div style="margin-bottom:6px;">{badge}<span style="color:#888;font-size:11px;">{p['source']}</span></div>
          <h3 style="margin:0 0 5px;font-size:14px;line-height:1.5;">
            <a href="{p['url']}" style="color:#1565c0;text-decoration:none;">{p['title']}</a>
          </h3>
          <p style="margin:0 0 10px;color:#777;font-size:11px;">{p['authors']}</p>
          <div style="background:#f5f5f5;padding:10px 12px;border-radius:4px;font-size:12px;color:#444;line-height:1.6;">
            💡 {p['why']}
          </div>
        </div>"""

    return f"""<html><body style="margin:0;padding:20px;background:#f0f2f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
<div style="max-width:580px;margin:0 auto;">
  <div style="background:#1565c0;color:#fff;padding:18px 22px;border-radius:8px 8px 0 0;">
    <h2 style="margin:0;font-size:17px;">📚 每日论文推荐 · {today}</h2>
    <p style="margin:4px 0 0;font-size:12px;opacity:.8;">UQ · AI4Health · AI4Omics | Powered by Claude</p>
  </div>
  <div style="padding:14px 0;">{cards}</div>
  <p style="text-align:center;color:#aaa;font-size:11px;margin-top:4px;">每日自动推送 · 为你的研究方向定制</p>
</div>
</body></html>"""


def send_email(papers: list[dict]):
    gmail_user = os.environ["GMAIL_USER"]
    gmail_pass = os.environ["GMAIL_APP_PASSWORD"]
    to_email   = os.environ.get("TO_EMAIL", gmail_user)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📚 论文推荐 {datetime.now().strftime('%m/%d')} · {sum(1 for p in papers if p['must_read'])} 篇精读"
    msg["From"]    = gmail_user
    msg["To"]      = to_email
    msg.attach(MIMEText(build_html(papers), "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(gmail_user, gmail_pass)
        s.sendmail(gmail_user, to_email, msg.as_string())
    print(f"✅ Sent to {to_email}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pool = []

    print("Fetching arXiv...")
    for q in ARXIV_QUERIES:
        pool.extend(fetch_arxiv(q))
        time.sleep(1)

    print("Fetching Semantic Scholar...")
    for q in SS_QUERIES:
        pool.extend(fetch_semantic_scholar(q))
        time.sleep(0.5)

    print("Fetching PubMed...")
    for q in PUBMED_QUERIES:
        pool.extend(fetch_pubmed(q))
        time.sleep(0.3)

    pool = deduplicate(pool)
    print(f"Candidates after dedup: {len(pool)}")

    if len(pool) > MAX_CANDIDATES:
        pool = random.sample(pool, MAX_CANDIDATES)

    print("Asking Claude to select...")
    selected = select_papers(pool)

    print("Sending email...")
    send_email(selected)


if __name__ == "__main__":
    main()
