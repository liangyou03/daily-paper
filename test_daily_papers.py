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
            "must_read_tag": "TODAY'S PICK",
            "one_liner_en": "A friendly one-sentence summary.",
            "one_liner_zh": "一句友好的中文摘要。",
            "what": "The question, data, method, and main result.",
            "innovation": "The concrete advance over prior work.",
            "learn": "What to study and reproduce for the dissertation.",
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
        self.assertIn("What this paper did", html)
        self.assertIn("What is genuinely new", html)
        self.assertIn("What you should learn", html)
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


if __name__ == "__main__":
    unittest.main()
