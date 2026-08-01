import sys
import types
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

sys.modules.setdefault("feedparser", types.SimpleNamespace(parse=lambda *_args, **_kwargs: None))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

import daily_papers


ROOT = Path(__file__).parent


class DissertationDigestConfigTests(unittest.TestCase):
    def test_searches_focus_on_dissertation_without_uq(self):
        search_text = " ".join(
            daily_papers.ARXIV_QUERIES
            + daily_papers.SS_CLASSIC_QUERIES
            + daily_papers.PUBMED_QUERIES
        ).lower()
        self.assertIn("microglia", search_text)
        self.assertIn("morphology", search_text)
        self.assertIn("spatial transcriptomics", search_text)
        self.assertIn("cell segmentation", search_text)
        self.assertIn("dapi", search_text)
        self.assertNotIn("uncertainty quantification", search_text)
        self.assertNotIn("conformal prediction", search_text)

    def test_domain_tagger_uses_dissertation_topics(self):
        cases = {
            "Microglial morphology and ramification in Alzheimer disease": "MicrogliaMorphology",
            "Spatial transcriptomics links tissue niches to cell states": "SpatialOmics",
            "Contrastive multimodal learning aligns images and gene expression": "MultimodalAI",
            "A benchmark of image encoders with donor-held-out evaluation": "Benchmark",
        }
        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assertEqual(
                    daily_papers.tag_domain({"title": title, "abstract": ""}), expected
                )

    def test_workflow_uses_deepseek_secret(self):
        workflow = (ROOT / ".github/workflows/daily_papers.yml").read_text()
        self.assertIn("DEEPSEEK_API_KEY", workflow)
        self.assertNotIn("GLM_API_KEY", workflow)

    def test_email_branding_matches_dissertation_digest(self):
        papers = [{
            "title": "A microglia paper",
            "url": "https://example.com",
            "source": "Nature Methods",
            "year": 2026,
            "authors": "A. Author",
            "domain": "MultimodalAI",
            "must_read": True,
            "must_read_tag": "READ TODAY",
            "reading_role": "READ TODAY",
            "biology_load": "Medium",
            "ml_load": "Low",
            "first_pass_minutes": 20,
            "one_liner_en": "A friendly one-sentence summary.",
            "one_liner_zh": "一句友好的中文摘要。",
            "paper_type": "Interdisciplinary",
            "data_types": [
                {"name": "Omics — spatial transcriptomics"},
                {"name": "Image — multiplex immunofluorescence"},
            ],
            "algorithms": [{
                "name": "GeoAdvAE",
                "purpose": "Aligns unpaired morphology and expression data.",
            }],
            "what": "The question, data, method, and main result.",
            "what_zh": "这篇论文研究了什么、用了什么数据和方法。",
            "innovation": "The concrete advance over prior work.",
            "innovation_zh": "它相对已有工作的主要推进。",
            "learn": "What to study and reproduce for the dissertation.",
            "learn_zh": "你应该重点学习和复现的部分。",
        }]
        glossary = [{
            "term_en": "Ramification",
            "term_zh": "分支化",
            "explanation_en": "How extensively a microglial cell branches.",
            "explanation_zh": "一句话说，就是小胶质细胞分支有多丰富。",
        }]
        biomedical_dictionary = [{
            "term_en": "Hippocampus",
            "term_zh": "海马体",
            "definition": "A brain region central to memory formation.",
        }]
        html = daily_papers.build_html(papers, glossary, biomedical_dictionary)
        self.assertIn("DeepSeek", html)
        self.assertNotIn("Microglia × Multimodal AI", html)
        self.assertIn("TERMS TO KNOW BEFORE YOU READ", html)
        self.assertIn("Ramification", html)
        self.assertIn("分支化", html)
        self.assertIn("At a glance", html)
        self.assertIn("A friendly one-sentence summary.", html)
        self.assertIn("一句友好的中文摘要。", html)
        self.assertIn("READ TODAY", html)
        self.assertIn("Biology load", html)
        self.assertIn("Medium", html)
        self.assertIn("ML load", html)
        self.assertIn("First-pass effort", html)
        self.assertIn("20 min", html)
        self.assertIn("Paper type", html)
        self.assertIn("Interdisciplinary", html)
        self.assertIn("Data types", html)
        self.assertIn("Omics — spatial transcriptomics", html)
        self.assertIn("Image — multiplex immunofluorescence", html)
        self.assertIn("Algorithms", html)
        self.assertIn("GeoAdvAE", html)
        self.assertIn("Aligns unpaired morphology and expression data.", html)
        self.assertNotIn("Paper type · 文章类型", html)
        self.assertNotIn("Data · 数据类型", html)
        self.assertNotIn("Algorithms · 算法", html)
        self.assertNotIn("交叉学科", html)
        self.assertIn("What this paper did", html)
        self.assertIn("这篇论文研究了什么、用了什么数据和方法。", html)
        self.assertIn("What is genuinely new", html)
        self.assertIn("它相对已有工作的主要推进。", html)
        self.assertIn("What you should learn", html)
        self.assertIn("你应该重点学习和复现的部分。", html)
        self.assertIn("BIOMEDICAL DICTIONARY", html)
        self.assertIn("Hippocampus", html)
        self.assertIn("海马体", html)
        self.assertIn("#cc785c", html)
        self.assertIn("#f7f4ed", html)
        self.assertIn("Georgia", html)
        self.assertNotIn("UQ", html)
        self.assertNotIn("GLM", html)

    def test_subject_describes_the_research_arc(self):
        subject = daily_papers.build_subject()
        self.assertIn("Dissertation Reading Brief", subject)
        self.assertIn("Segmentation to Cell State", subject)
        self.assertNotIn("篇", subject)

    def test_glossary_terms_are_never_repeated(self):
        glossary = [
            {"term_en": "  Spatial   transcriptomics ", "term_zh": "空间转录组"},
            {"term_en": "Ramification", "term_zh": "分支化"},
            {"term_en": "ramification", "term_zh": "重复项"},
        ]
        filtered = daily_papers.filter_new_glossary_terms(
            glossary, {"spatial transcriptomics"}
        )
        self.assertEqual([term["term_en"] for term in filtered], ["Ramification"])

    def test_glossary_history_round_trip(self):
        terms = [{"term_en": "Ramification", "term_zh": "分支化"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "glossary_history.md"
            daily_papers.save_glossary_history(terms, path)
            self.assertEqual(daily_papers.load_glossary_history(path), {"ramification"})

    def test_deepseek_selection_disables_thinking_for_json(self):
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = types.SimpleNamespace(
                    content='{"papers":[{"pool":"recent","index":1,"domain":"SpatialOmics","must_read_tag":"","why":"relevant"}]}'
                )
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )
        paper = {
            "title": "Microglia in spatial transcriptomics",
            "abstract": "Abstract",
            "authors": "Author",
            "source": "Journal",
            "domain": "SpatialOmics",
            "url": "https://example.com",
            "year": 2026,
        }

        with patch.object(daily_papers, "OpenAI", return_value=fake_client), patch.dict(
            daily_papers.os.environ, {"DEEPSEEK_API_KEY": "test-key"}
        ):
            daily_papers.select_papers([paper], [], [])

        self.assertEqual(captured.get("extra_body"), {"thinking": {"type": "disabled"}})

    def test_selection_prompt_requests_three_papers(self):
        source = (ROOT / "daily_papers.py").read_text()
        self.assertIn("Select exactly 3 papers total", source)
        self.assertIn("2 from the RECENT pool and 1 from the CLASSIC pool", source)
        self.assertIn("Select exactly 3 papers from the RECENT pool", source)
        self.assertNotIn("Select exactly 2 papers", source)

    def test_selection_prompt_requests_brief_bilingual_sections(self):
        source = (ROOT / "daily_papers.py").read_text()
        self.assertIn("what_zh", source)
        self.assertIn("innovation_zh", source)
        self.assertIn("learn_zh", source)
        self.assertIn("1–2 short sentences", source)

    def test_selection_prompt_extracts_structured_paper_metadata(self):
        source = (ROOT / "daily_papers.py").read_text()
        self.assertIn('"paper_type": "Biology" or "Deep Learning" or "Interdisciplinary"', source)
        self.assertIn('"data_types":', source)
        self.assertIn('"algorithms":', source)
        self.assertNotIn('"paper_type_zh":', source)
        self.assertNotIn('"purpose_zh":', source)
        self.assertIn("actual study data", source)
        self.assertIn("do not infer a specific algorithm", source)

    def test_dissertation_relevance_prefers_brain_biology(self):
        brain = {
            "title": "Microglial morphology near amyloid plaques in Alzheimer's cortex",
            "abstract": "Paired immunofluorescence and spatial transcriptomics in human brain.",
        }
        distant = {
            "title": "Multimodal survival prediction in colorectal cancer",
            "abstract": "A generic clinical model for tumor prognosis.",
        }
        self.assertGreater(
            daily_papers.score_dissertation_relevance(brain),
            daily_papers.score_dissertation_relevance(distant),
        )

    def test_journal_qc_removes_mdpi_venues(self):
        papers = [
            {"title": "Paper A", "source": "Cells"},
            {"title": "Paper B", "source": "Brain Sciences"},
            {"title": "Paper C", "source": "Nature Methods"},
        ]
        filtered = daily_papers.filter_journal_quality(papers)
        self.assertEqual([paper["source"] for paper in filtered], ["Nature Methods"])

    def test_journal_qc_prefers_field_leading_venues(self):
        self.assertGreater(
            daily_papers.journal_quality_score({"source": "Nature Methods"}),
            daily_papers.journal_quality_score({"source": "arXiv"}),
        )

    def test_selection_prompt_enforces_journal_qc(self):
        source = (ROOT / "daily_papers.py").read_text()
        self.assertIn("NEVER select papers from MDPI journals", source)
        self.assertIn('"venue"', source)

    def test_pubmed_parser_extracts_concise_article_metadata(self):
        xml = """
        <PubmedArticleSet><PubmedArticle><MedlineCitation>
          <PMID>123</PMID><Article>
            <Journal><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue>
              <Title>Neuron</Title></Journal>
            <ArticleTitle>A concise microglia study</ArticleTitle>
            <Pagination><MedlinePgn>101-106</MedlinePgn></Pagination>
            <Abstract><AbstractText>Short abstract.</AbstractText></Abstract>
            <AuthorList><Author><LastName>Li</LastName><ForeName>Liang</ForeName></Author></AuthorList>
            <PublicationTypeList><PublicationType>Brief Report</PublicationType></PublicationTypeList>
          </Article>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>
        """
        paper = daily_papers._parse_pubmed_xml(["123"], xml)[0]
        self.assertEqual(paper["pages"], "101-106")
        self.assertEqual(paper["publication_types"], ["Brief Report"])

    def test_short_formats_are_eligible_but_not_prioritized(self):
        source = (ROOT / "daily_papers.py").read_text()
        self.assertIn("are fully eligible", source)
        self.assertIn("do not prefer them", source)
        self.assertNotIn("article_conciseness_score", source)

    def test_search_and_prompt_allow_relevant_computational_methods(self):
        source = (ROOT / "daily_papers.py").read_text()
        search_text = " ".join(daily_papers.ARXIV_QUERIES + daily_papers.PUBMED_QUERIES).lower()
        self.assertIn("computational biology microglia", search_text)
        self.assertIn("deep learning microglia", search_text)
        self.assertIn("Brief Report", source)
        self.assertIn("must remain directly relevant", source)

    def test_top_journal_search_includes_computational_biology_venues(self):
        journal_query = daily_papers.JOURNAL_FILTER.lower()
        self.assertIn("plos computational biology", journal_query)
        self.assertIn("medical image analysis", journal_query)
        self.assertIn("patterns", journal_query)

    def test_reading_roles_put_required_paper_first(self):
        papers = [
            {"title": "Later", "reading_role": "SAVE FOR LATER"},
            {"title": "Method", "reading_role": "OPTIONAL METHOD PAPER"},
            {"title": "Today", "reading_role": "READ TODAY"},
        ]
        ordered = daily_papers.order_papers_for_email(papers)
        self.assertEqual([paper["title"] for paper in ordered], ["Today", "Method", "Later"])

    def test_selection_prompt_requests_habit_friendly_reading_fields(self):
        source = (ROOT / "daily_papers.py").read_text()
        self.assertIn('"reading_role":', source)
        self.assertIn('"biology_load":', source)
        self.assertIn('"ml_load":', source)
        self.assertIn('"first_pass_minutes":', source)
        self.assertIn("one required paper", source)
        self.assertNotIn('"reading_goal_en":', source)
        self.assertNotIn("15-MINUTE READING PLAN", source)

    def test_selection_prefers_medium_or_high_ml_difficulty(self):
        source = (ROOT / "daily_papers.py").read_text()
        self.assertIn("Prefer ML load Medium or High", source)
        self.assertIn(
            "READ TODAY and OPTIONAL METHOD PAPER should both have ML load Medium or High",
            source,
        )
        self.assertIn("A Low-ML paper is acceptable only", source)


if __name__ == "__main__":
    unittest.main()
