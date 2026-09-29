import hashlib
import math
import numpy as np
import pytest
import torch


def test_01(api):
    assert api.normalize_answer("  The CAT!  ") == "the cat"
    assert api.normalize_answer("a-b") == "ab" and api.normalize_answer("a b") == "a b"
    assert api.normalize_answer("Straße") == "strasse"


def test_02(api):
    assert api.exact_match("Paris!", ["paris", "city of paris"]) == 1.0
    assert api.exact_match("the paris", ["paris"]) == 0.0
    with pytest.raises(ValueError):
        api.exact_match("", [])


def test_03(api, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(5, 7, generator=g)
    ids = [1, 2, 3, 4, 5]
    expected = sum(float(x[i - 1].log_softmax(0)[ids[i]]) for i in range(2, 5))
    assert api.continuation_logp(x, ids, 2) == pytest.approx(expected)
    altered = x.clone()
    altered[0] += torch.arange(7) * 100
    altered[-1] *= 100
    assert api.continuation_logp(altered, ids, 2) == pytest.approx(expected)
    with pytest.raises(ValueError):
        api.continuation_logp(x, ids, 0)


def test_04(api):
    assert api.rank_choices([-2.0, -3.0], [1, 3]) == 0
    assert api.rank_choices([-2.0, -3.0], [1, 3], True) == 1
    assert api.rank_choices([-1.0, -1.0], [1, 1]) == 0
    with pytest.raises(ValueError):
        api.rank_choices([-1], [0])


def test_05(api):
    assert api.perplexity([2.0, 12.0], [1, 3]) == pytest.approx(math.exp(3.5))
    assert api.perplexity([0.0], [7]) == 1
    with pytest.raises(ValueError):
        api.perplexity([1.0], [0])


def test_06(api):
    m = api.confusion_matrix([0, 0, 1, 2], [0, 1, 1, 0], 3)
    np.testing.assert_array_equal(m, [[1, 1, 0], [0, 1, 0], [1, 0, 0]])
    assert m.dtype == np.int64
    with pytest.raises(ValueError):
        api.confusion_matrix([3], [0], 3)


def test_07(api):
    r = api.classification_metrics([[1, 1, 0], [0, 1, 0], [1, 0, 0]])
    assert r["accuracy"] == 0.5 and r["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 3)
    assert r["per_class_f1"] == pytest.approx([0.5, 2 / 3, 0])
    assert api.classification_metrics(np.zeros((2, 2)))["macro_f1"] == 0


def test_08(api, seed):
    a = [0, 1, 0, 1]
    b = [1, 1, 1, 1]
    r = api.paired_bootstrap(a, b, seed, resamples=100)
    delta = np.array(b) - a
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, 4, size=(100, 4))].mean(1)
    assert r == dict(
        delta=0.5, low=float(np.quantile(samples, 0.025)), high=float(np.quantile(samples, 0.975))
    )
    zero = api.paired_bootstrap(a, a, seed)
    assert zero == dict(delta=0, low=0, high=0)
    constant = api.paired_bootstrap(a, np.array(a) + 0.2, seed)
    assert constant["low"] == pytest.approx(0.2) and constant["high"] == pytest.approx(0.2)


def test_09(api, seed):
    a = [dict(id="a", score=0), dict(id="b", score=1)]
    b = [dict(id="b", score=0), dict(id="a", score=1)]
    assert api.compare_by_id(a, b, seed) == api.compare_by_id(a[::-1], b[::-1], seed)
    assert api.compare_by_id(a, b, seed)["delta"] == 0
    with pytest.raises(ValueError):
        api.compare_by_id(a, b[:1])
    with pytest.raises(ValueError):
        api.compare_by_id(a, [b[0], b[0]])


def test_10(api):
    rows = [dict(group="large", correct=True)] * 9 + [dict(group="small", correct=False)]
    r = api.slice_metrics(rows)
    assert r["micro_accuracy"] == 0.9 and r["macro_accuracy"] == 0.5 and r["worst_accuracy"] == 0
    assert r["groups"]["small"] == dict(n=1, accuracy=0)


def test_11(api):
    assert api.calibration_error([0.1, 0.5, 1.0], [False, True, True], 2) == pytest.approx(
        (0.1 + 2 * 0.25) / 3
    )
    assert api.calibration_error([0, 1], [False, True]) == 0
    with pytest.raises(ValueError):
        api.calibration_error([1.1], [True])


def test_12(api):
    r = api.latency_metrics(1, [1.5, 1.7, 2.0])
    assert (
        r["ttft"] == 0.5
        and r["inter_token_seconds"] == pytest.approx([0.2, 0.3])
        and r["decode_tokens_per_second"] == 4
    )
    assert api.latency_metrics(0, [1])["decode_tokens_per_second"] is None
    with pytest.raises(ValueError):
        api.latency_metrics(0, [2, 1])


def test_13(api):
    seen = []

    def predict(prompt):
        seen.append(prompt)
        return "yes"

    cases = [
        dict(id="a", prompt="question 1", acceptable=["yes"]),
        dict(id="b", prompt="question 2", acceptable=["no"]),
    ]
    r = api.evaluate(predict, cases, "a" * 64, "b" * 64)
    assert (
        seen == ["question 1", "question 2"]
        and r["metrics"] == dict(accuracy=0.5, n=2)
        and r["format"] == "toyeval-v1"
    )
    assert r["model_sha256"] == "a" * 64 and [x["id"] for x in r["cases"]] == ["a", "b"]
    with pytest.raises(ValueError):
        api.evaluate(predict, [cases[0], cases[0]], "a" * 64, "b" * 64)

    def bad(_):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError):
        api.evaluate(bad, cases, "a" * 64, "b" * 64)


def test_14(api, tmp_path):
    path = tmp_path / "eval.json"
    report = {"z": 2, "a": "é"}
    digest = api.write_report(path, report)
    assert (
        path.read_bytes() == '{"a":"é","z":2}\n'.encode()
        and digest == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        api.write_report(path, {})
    with pytest.raises(ValueError):
        api.write_report(tmp_path / "bad.json", {"n": float("nan")})
    assert not (tmp_path / "bad.json").exists()
