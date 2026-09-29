import hashlib
import json
import random
import pytest


def test_01(api):
    assert api.normalize("  ＨＥＬＬＯ\tStraße\n") == "hello strasse"
    assert api.normalize("e\u0301") == api.normalize("é")
    assert api.normalize(" \n ") == ""


def test_02(api):
    assert api.document_id(" Hello ") == hashlib.sha256(b"hello").hexdigest()
    assert api.document_id("Ａ") == api.document_id("a")
    assert api.document_id("ab") != api.document_id("a b")


def test_03(api):
    assert api.quality_filter("one two three")
    assert not api.quality_filter("a a a b") and not api.quality_filter("")
    assert api.quality_filter("a a b c") and not api.quality_filter("one two")


def test_04(api):
    records = [
        {"id": "a", "text": "Hello world"},
        {"id": "b", "text": " HELLO  world "},
        {"id": "c", "text": "Different"},
    ]
    result = api.deduplicate(records)
    assert [r["id"] for r in result] == ["a", "c"]
    result[0]["text"] = "changed"
    assert records[0]["text"] == "Hello world"


def test_05(api):
    assert api.shingles("A b c d", 2) == {("a", "b"), ("b", "c"), ("c", "d")}
    assert api.shingles("a", 2) == set()
    with pytest.raises(ValueError):
        api.shingles("a", 0)


def test_06(api):
    records = [
        dict(id="a", text="a b c"),
        dict(id="b", text="b c d"),
        dict(id="c", text="c d e"),
        dict(id="d", text="short"),
        dict(id="e", text="other"),
    ]
    assert api.duplicate_components(records, threshold=1 / 3, n=2) == [
        ["a", "b", "c"],
        ["d"],
        ["e"],
    ]
    with pytest.raises(ValueError):
        api.duplicate_components([records[0], records[0]])
    assert api.duplicate_components([dict(id="x", text="short"), dict(id="y", text="SHORT")]) == [
        ["x", "y"]
    ]


def test_07(api):
    groups = [["a", "b"], ["c"], ["d", "e"]]
    a = api.split_groups(groups)
    assert a == api.split_groups([["e", "d"], ["c"], ["b", "a"]]) and a["a"] == a["b"]
    assert set(api.split_groups(groups, 1).values()) == {"validation"}
    assert set(api.split_groups(groups, 0).values()) == {"train"}
    for g in groups:
        value = int(hashlib.sha256(("course:" + min(g)).encode()).hexdigest(), 16) / 2**256
        assert a[g[0]] == ("validation" if value < 0.2 else "train")
    with pytest.raises(ValueError):
        api.split_groups([["a"], ["a"]])


def test_08(api):
    r = [
        dict(id="a", text="the hidden test answer"),
        dict(id="b", text="fresh original training sample"),
        dict(id="c", text="short"),
    ]
    assert [x["id"] for x in api.decontaminate(r, ["hidden test answer", "SHORT"])] == ["b"]
    assert api.decontaminate(r, []) == r


def test_09(api, seed):
    src = {"a": [dict(id=1)], "b": [dict(id=2), dict(id=3)]}
    weights = {"a": 0.2, "b": 0.8}
    before = random.getstate()
    a = api.sample_mixture(src, weights, 100, seed)
    assert random.getstate() == before and a == api.sample_mixture(src, weights, 100, seed)
    assert a == api.sample_mixture(dict(reversed(list(src.items()))), weights, 100, seed)
    assert len(a) == 100 and {r["source"] for r in a} == {"a", "b"} and "source" not in src["a"][0]
    assert all(x["source"] == "b" for x in api.sample_mixture(src, {"a": 0, "b": 1}, 10, seed))
    with pytest.raises(ValueError):
        api.sample_mixture(src, {"a": 0, "b": 0}, 2)


def test_10(api):
    docs = [[256, 1, 257], [], [256, 2, 3, 257], [256, 4, 257]]
    saved = [d[:] for d in docs]
    blocks = api.pack_documents(docs, 8)
    assert docs == saved and len(blocks) == 2
    assert blocks[0] == dict(
        input_ids=[256, 1, 257, 256, 2, 3, 257, 0],
        segment_ids=[0, 0, 0, 2, 2, 2, 2, -1],
        position_ids=[0, 1, 2, 0, 1, 2, 3, 0],
        loss_mask=[False, True, True, False, True, True, True, False],
    )
    assert api.pack_documents([], 4) == []
    with pytest.raises(ValueError):
        api.pack_documents([[1, 2, 3]], 2)


def test_11(api):
    assert api.attention_mask([0, 0, 1, -1]) == [
        [True, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, False, False],
    ]
    assert api.attention_mask([]) == []


def test_12(api, tmp_path):
    records = [{"id": "x", "text": "é"}]
    m = api.write_dataset(tmp_path, records)
    body = (tmp_path / "records.jsonl").read_bytes()
    assert body == '{"id":"x","text":"é"}\n'.encode()
    assert m == dict(
        format="toydata-v1",
        file="records.jsonl",
        sha256=hashlib.sha256(body).hexdigest(),
        records=1,
    )
    assert json.loads((tmp_path / "manifest.json").read_text()) == m
    with pytest.raises(FileExistsError):
        api.write_dataset(tmp_path, [])
    assert (tmp_path / "records.jsonl").read_bytes() == body


def test_13(api, tmp_path):
    records = [
        dict(id="a", text="A fresh original document"),
        dict(id="b", text="a fresh ORIGINAL document"),
        dict(id="c", text="secret held out answer"),
        dict(id="d", text="x x x"),
        dict(id="e", text="a fresh original document extended"),
    ]
    m = api.prepare_dataset(records, ["secret held out answer"], tmp_path)
    out = [json.loads(x) for x in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert m["records"] == 2 and [r["id"] for r in out] == ["a", "e"]
    assert out[0]["split"] == out[1]["split"] and all(len(r["content_sha256"]) == 64 for r in out)
    assert "split" not in records[0]
    with pytest.raises(ValueError):
        api.prepare_dataset([records[0], records[0]], [], tmp_path / "bad")
