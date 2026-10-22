"""Local source identity primitives; no GPU, SDK or outer-cache coverage claimed."""

import json

import pytest

from triton_msl.backend import _cache_contract as contract


def test_package_content_is_relocatable_and_content_sensitive(tmp_path):
    a, b = tmp_path / "first checkout", tmp_path / "wheel installation"
    for root in (a, b):
        root.mkdir()
        (root / "__init__.py").write_text("VERSION = 1\n")
        (root / "native.so").write_bytes(b"native implementation one")
    assert contract._package_content(a) == contract._package_content(b)
    (b / "native.so").write_bytes(b"native implementation two")
    assert contract._package_content(a) != contract._package_content(b)


def test_missing_package_does_not_get_an_unknown_identity(tmp_path):
    with pytest.raises(RuntimeError, match="incomplete"):
        contract._package_content(tmp_path)


def test_content_and_options_are_separate_key_fields():
    assert contract.source_key("ab", "c") != contract.source_key("a", "bc")


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_effective_policy_uses_memoized_stamp_encoder(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TRITON_MSL_CPP_SCALAR_CFG", raising=False)
    else:
        monkeypatch.setenv("TRITON_MSL_CPP_SCALAR_CFG", value)
    policy = contract.effective_policy()
    # Use the real producer, not a policy synthesized from the encoder's keys:
    # adding a policy flag must not silently bypass the known-stamp fast path.
    assert policy.keys() == contract._STAMP_POLICY_FIELDS
    assert policy["CPP_SCALAR_CFG"] is (value == "1")
    source = {
        "schema": contract.SOURCE_SCHEMA,
        "implementation": "controlled-package",
        "frameworks": "controlled-framework",
        "label": "controlled-label",
        "policy": policy,
    }
    expected = json.dumps(
        {"schema": 1, "source": source, "toolchain": "controlled-toolchain"},
        sort_keys=True,
        separators=(",", ":"),
    )
    contract._encode_known_stamp.cache_clear()
    try:
        assert contract._encode_execution(source, "controlled-toolchain") == expected
        assert contract._encode_execution(source, "controlled-toolchain") == expected
        info = contract._encode_known_stamp.cache_info()
        assert (info.misses, info.hits) == (1, 1)
    finally:
        contract._encode_known_stamp.cache_clear()


@pytest.mark.parametrize("value", [None, "1", "true", "TRUE", "False"])
def test_mept_truth_rule_matches_the_lowerer(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TRITON_MSL_MEPT", raising=False)
    else:
        monkeypatch.setenv("TRITON_MSL_MEPT", value)
    assert contract.effective_policy()["MEPT"] is True
    monkeypatch.setenv("TRITON_MSL_MEPT", "0")
    assert contract.effective_policy()["MEPT"] is False


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("True", True), ("TRUE", False), ("0", False)])
def test_half_accum_truth_rule_matches_the_lowerer(monkeypatch, value, expected):
    monkeypatch.setenv("TRITON_MSL_FA_HALF_ACCUM", value)
    assert contract.effective_policy()["FA_HALF_ACCUM"] is expected


@pytest.mark.parametrize(
    "damage",
    [
        "old_schema",
        "bool_schema",
        "missing_key",
        "wrong_implementation",
        "wrong_policy",
        "wrong_source",
        "wrong_metadata",
        "non_dict",
    ],
)
def test_bound_metadata_record_rejects_invalid_envelopes(tmp_path, damage):
    source = "kernel void source() {}"
    path = tmp_path / "record.json"
    record = json.loads(contract.metadata_record(source, {"block_size": 64}, "key"))
    if damage == "old_schema":
        record["schema"] = contract.SOURCE_SCHEMA - 1
    elif damage == "bool_schema":
        record["schema"] = True
    elif damage == "missing_key":
        record.pop("key")
    elif damage == "wrong_implementation":
        record["contract"]["implementation"] = "old"
    elif damage == "wrong_policy":
        record["contract"]["policy"]["MEPT"] = not record["contract"]["policy"]["MEPT"]
    elif damage == "wrong_source":
        record["source_sha256"] = "old"
    elif damage == "wrong_metadata":
        record["metadata"]["block_size"] = 1024
    else:
        record = []
    path.write_text(json.dumps(record))
    assert contract.read_metadata(source, path, "key") is None


def test_unserializable_metadata_is_not_silently_omitted():
    with pytest.raises(TypeError):
        contract.metadata_record("source", {"descriptor": object()}, "key")
