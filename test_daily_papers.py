import sys
import types
import unittest
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
        html = daily_papers.build_html([{
            "title": "A microglia paper",
            "url": "https://example.com",
            "source": "Nature Methods",
            "year": 2026,
            "authors": "A. Author",
            "domain": "MultimodalAI",
            "must_read": True,
            "must_read_tag": "⭐ 今日精读",
            "one_liner": "一句话看懂这篇论文。",
            "what": "研究问题、数据、方法与主要结果。",
            "innovation": "相较既有工作的关键增量。",
            "learn": "值得复现的方法和阅读重点。",
        }])
        self.assertIn("DeepSeek", html)
        self.assertIn("Microglia", html)
        self.assertIn("一句话看懂", html)
        self.assertIn("论文做了什么", html)
        self.assertIn("创新在哪里", html)
        self.assertIn("你应该学什么", html)
        self.assertNotIn("UQ", html)
        self.assertNotIn("GLM", html)

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
