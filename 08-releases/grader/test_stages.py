from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pytest


def report(scores, model="a" * 64, data="d" * 64):
    return dict(
        format="toyeval-v1",
        model_sha256=model,
        dataset_sha256=data,
        metrics=dict(n=len(scores), accuracy=sum(scores) / len(scores)),
        cases=[dict(id=str(i), prediction="x", score=s) for i, s in enumerate(scores)],
    )


def test_01(api):
    assert api.canonical_json({"z": 2, "a": "é"}) == '{"a":"é","z":2}\n'.encode()
    with pytest.raises(ValueError):
        api.canonical_json({"bad": float("nan")})


def test_02(api, tmp_path):
    p = tmp_path / "blob"
    body = b"abc" * 800000
    p.write_bytes(body)
    assert api.file_digest(p) == hashlib.sha256(body).hexdigest()
    p.write_bytes(b"")
    assert api.file_digest(p) == hashlib.sha256(b"").hexdigest()


def test_03(api, tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "a").write_text("ok")
    outside = tmp_path / "secret"
    outside.write_text("x")
    (root / "link").symlink_to(outside)
    assert api.safe_artifact_path(root, "a") == (root / "a").resolve()
    for bad in ["../secret", str(outside), "link", "missing", ""]:
        with pytest.raises(ValueError):
            api.safe_artifact_path(root, bad)


def test_04(api, tmp_path):
    (tmp_path / "b").write_bytes(b"b")
    (tmp_path / "a").write_bytes(b"a")
    metadata = {"seed": 1}
    m = api.build_manifest(tmp_path, ["b", "a"], metadata)
    assert m["format"] == "toyrelease-v1" and [r["path"] for r in m["files"]] == ["a", "b"]
    assert m["files"][0] == dict(path="a", sha256=hashlib.sha256(b"a").hexdigest(), size_bytes=1)
    m["metadata"]["seed"] = 9
    assert metadata["seed"] == 1
    with pytest.raises(ValueError):
        api.build_manifest(tmp_path, ["a", "a"], {})


def test_05(api, tmp_path):
    p = tmp_path / "a"
    p.write_text("first")
    m = api.build_manifest(tmp_path, ["a"], {})
    assert api.verify_manifest(tmp_path, m) is True
    p.write_text("other")
    with pytest.raises(ValueError):
        api.verify_manifest(tmp_path, m)


def test_06(api, tmp_path):
    src = tmp_path / "model"
    src.write_bytes(b"weights")
    store = tmp_path / "store"
    digest = api.register_artifact(store, src)
    assert (store / digest).read_bytes() == b"weights" and src.read_bytes() == b"weights"
    assert api.register_artifact(store, src) == digest and len(list(store.iterdir())) == 1
    (store / digest).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        api.register_artifact(store, src)


def test_07(api):
    r = report([1, 0])
    assert api.validate_evaluation(r, "a" * 64, "d" * 64) is True
    r["metrics"]["accuracy"] = 1
    with pytest.raises(ValueError):
        api.validate_evaluation(r, "a" * 64, "d" * 64)
    with pytest.raises(ValueError):
        api.validate_evaluation(report([1]), "b" * 64, "d" * 64)
    r = report([1, 0])
    r["cases"][1]["id"] = "0"
    with pytest.raises(ValueError):
        api.validate_evaluation(r, "a" * 64, "d" * 64)


def test_08(api):
    a = report([1, 1, 1, 1, 0])
    b = report([1, 1, 1, 0, 0], model="b" * 64)
    assert api.release_gate(a, b, min_accuracy=0.7, max_drop=0.1, min_cases=10) == dict(
        eligible=False, reasons=["too_few_cases", "below_floor", "regression"]
    )
    assert api.release_gate(a, a, min_accuracy=0.8, max_drop=0, min_cases=5)["eligible"]
    with pytest.raises(ValueError):
        api.release_gate(a, report([1], data="c" * 64))


def test_09(api):
    assert api.route_request("id", 0) == "baseline" and api.route_request("id", 1) == "candidate"
    for i in range(30):
        key = str(i)
        ratio = int(hashlib.sha256(("course:" + key).encode()).hexdigest(), 16) / 2**256
        assert api.route_request(key, 0.3) == ("candidate" if ratio < 0.3 else "baseline")
    with pytest.raises(ValueError):
        api.route_request("id", 2)


def test_10(api):
    records = [dict(status="error" if i == 19 else "ok", latency_seconds=i / 10) for i in range(20)]
    assert api.summarize_requests(records) == dict(n=20, error_rate=0.05, p95_seconds=1.8)
    with pytest.raises(ValueError):
        api.summarize_requests([dict(status="ok", latency_seconds=-1)])


def test_11(api):
    assert api.rollback_decision(dict(n=19, error_rate=1, p95_seconds=9)) == "wait"
    assert api.rollback_decision(dict(n=20, error_rate=0.05, p95_seconds=2)) == "keep"
    assert api.rollback_decision(dict(n=20, error_rate=0.1, p95_seconds=1)) == "rollback"
    assert api.rollback_decision(dict(n=20, error_rate=0, p95_seconds=3)) == "rollback"


def test_12(api, tmp_path):
    p = tmp_path / "active.json"
    assert api.promote(p, "a" * 64, None) == dict(current="a" * 64, previous=None)
    assert api.promote(p, "a" * 64, "a" * 64) == dict(current="a" * 64, previous=None)

    def attempt(d):
        try:
            api.promote(p, d, "a" * 64)
            return "ok"
        except RuntimeError:
            return "conflict"

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(attempt, ["b" * 64, "c" * 64]))
    assert sorted(results) == ["conflict", "ok"]
    r = json.loads(p.read_text())
    assert r["previous"] == "a" * 64 and r["current"] in ["b" * 64, "c" * 64]


def test_13(api, tmp_path):
    p = tmp_path / "active.json"
    api.promote(p, "a" * 64, None)
    with pytest.raises(ValueError):
        api.rollback(p, "a" * 64)
    api.promote(p, "b" * 64, "a" * 64)
    assert api.rollback(p, "b" * 64) == dict(current="a" * 64, previous="b" * 64)
    with pytest.raises(RuntimeError):
        api.rollback(p, "b" * 64)


def test_14(api, tmp_path):
    model = tmp_path / "model"
    model.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    ev = tmp_path / "eval.json"
    baseline = report([1, 1, 0, 0])
    pointer = tmp_path / "active.json"
    store = tmp_path / "objects"
    ev.write_text(json.dumps(report([0, 0, 0, 0], model=digest)))
    with pytest.raises(ValueError):
        api.release_model(model, ev, baseline, store, pointer, None, min_accuracy=0.5, min_cases=4)
    assert not pointer.exists() and not store.exists()
    ev.write_text(json.dumps(report([1, 1, 1, 1], model=digest)))
    r = api.release_model(model, ev, baseline, store, pointer, None, min_accuracy=0.5, min_cases=4)
    assert r["model_sha256"] == digest and json.loads(pointer.read_text())["current"] == digest
    assert (store / r["report_sha256"]).read_bytes() == ev.read_bytes()
    ev.write_text(json.dumps(report([1, 1, 1, 1], model="f" * 64)))
    with pytest.raises(ValueError):
        api.release_model(
            model, ev, baseline, store, pointer, digest, min_accuracy=0.5, min_cases=4
        )
