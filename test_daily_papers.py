import sys
import types
import unittest
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
        html = daily_papers.build_html([])
        self.assertIn("DeepSeek", html)
        self.assertIn("Microglia", html)
        self.assertNotIn("UQ", html)
        self.assertNotIn("GLM", html)


if __name__ == "__main__":
    unittest.main()
