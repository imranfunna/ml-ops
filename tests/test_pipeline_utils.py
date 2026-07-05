"""Unit tests for pipeline_utils. Runs in CI without Spark."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from pipeline_utils import (
    align_distributions,
    detect_language,
    mask_pii,
    normalize_text,
    population_stability_index,
    to_probabilities,
    top_k_cosine,
)


class TestNormalizeText:
    def test_none_returns_empty(self):
        assert normalize_text(None) == ""

    def test_url_stripped(self):
        assert "<url>" in normalize_text("check https://foo.com now")

    def test_mention_stripped(self):
        assert "<user>" in normalize_text("hey @support help")

    def test_template_replaced(self):
        out = normalize_text("cancel {{Order Number}} please")
        assert "<order_number>" in out
        assert "{{" not in out

    def test_whitespace_collapsed(self):
        assert normalize_text("a    b\t\tc") == "a b c"


class TestMaskPII:
    def test_email_masked(self):
        assert mask_pii("mail me at foo@bar.com") == "mail me at <email>"

    def test_phone_masked(self):
        assert "<phone>" in mask_pii("call +31 6 12345678 asap")

    def test_order_masked(self):
        assert "<order_ref>" in mask_pii("issue with order #12345")

    def test_no_pii_untouched(self):
        assert mask_pii("hello world") == "hello world"


class TestDetectLanguage:
    def test_empty(self):
        assert detect_language("") == "unknown"

    def test_english(self):
        assert detect_language(" the quick brown fox for you ") == "en"

    def test_dutch(self):
        assert detect_language(" ik heb een probleem met de app niet ") == "nl"

    def test_default_english(self):
        assert detect_language("xyz") == "en"


class TestPSI:
    def test_identical_distributions_zero(self):
        p = [0.5, 0.5]
        assert population_stability_index(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_shift_positive(self):
        assert population_stability_index([0.5, 0.5], [0.9, 0.1]) > 0.25

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            population_stability_index([0.5, 0.5], [1.0])


class TestDistributionHelpers:
    def test_to_probabilities(self):
        assert to_probabilities({"a": 3, "b": 1}) == {"a": 0.75, "b": 0.25}

    def test_to_probabilities_empty(self):
        assert to_probabilities({"a": 0, "b": 0}) == {"a": 0.0, "b": 0.0}

    def test_align(self):
        e, a = align_distributions({"x": 0.6, "y": 0.4}, {"y": 0.5, "z": 0.5})
        assert len(e) == len(a) == 3
        assert sum(e) == pytest.approx(1.0)
        assert sum(a) == pytest.approx(1.0)


class TestTopK:
    def test_topk(self):
        q = [1.0, 0.0]
        m = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        assert top_k_cosine(q, m, k=2) == [0, 2]
