"""Tests for the safetensors reader and the bfloat16 conversion.

These build their own files rather than leaning on the downloaded model, so
they run in CI where no weights exist. The one test that does use the real
model is skipped when it is absent.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from nanoinfer.safetensors import SafeTensors, bf16_to_f32, f32_to_bf16

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict | None = None) -> Path:
    """Build a minimal safetensors file by hand, to test the reader against."""
    dtype_names = {
        np.dtype("<f4"): "F32",
        np.dtype("<f8"): "F64",
        np.dtype("<i4"): "I32",
        np.dtype("<i8"): "I64",
        np.dtype("<u2"): "BF16",
        np.dtype(np.int8): "I8",
        np.dtype(np.uint8): "U8",
        np.dtype(np.bool_): "BOOL",
    }

    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata

    blob = bytearray()
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        begin = len(blob)
        blob.extend(arr.tobytes())
        header[name] = {
            "dtype": dtype_names[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [begin, len(blob)],
        }

    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(blob))
    return path


# -- bfloat16 --------------------------------------------------------------


def test_bf16_known_bit_patterns():
    """Hand-checked values: bf16 is the top 16 bits of the f32 pattern."""
    cases = {
        0x0000: 0.0,
        0x8000: -0.0,
        0x3F80: 1.0,
        0xBF80: -1.0,
        0x4000: 2.0,
        0x3F00: 0.5,
        0x4049: 3.140625,       # nearest bf16 to pi
        0x7F80: float("inf"),
        0xFF80: float("-inf"),
    }
    raw = np.array(list(cases), dtype="<u2")
    got = bf16_to_f32(raw)
    np.testing.assert_array_equal(got, np.array(list(cases.values()), dtype=np.float32))


def test_bf16_nan_survives():
    raw = np.array([0x7FC0], dtype="<u2")
    assert np.isnan(bf16_to_f32(raw)[0])


def test_bf16_round_trip_is_exact_for_representable_values():
    """Any value that came from bf16 must survive widen -> narrow unchanged."""
    raw = np.arange(0, 0x10000, dtype="<u2")
    widened = bf16_to_f32(raw)
    finite = np.isfinite(widened)
    back = f32_to_bf16(widened[finite])
    np.testing.assert_array_equal(back, raw[finite])


def test_bf16_widening_never_changes_the_value():
    """Widening adds zero bits, so the float32 result is exactly equal."""
    raw = np.random.default_rng(0).integers(0, 0x7F80, size=4096, dtype=np.uint16).astype("<u2")
    widened = bf16_to_f32(raw)
    # Narrowing a value that is already bf16-exact must be the identity.
    np.testing.assert_array_equal(f32_to_bf16(widened), raw)


def test_bf16_rejects_wrong_dtype():
    with pytest.raises(TypeError):
        bf16_to_f32(np.zeros(4, dtype=np.float32))


def test_bf16_preserves_shape():
    raw = np.zeros((3, 5, 7), dtype="<u2")
    assert bf16_to_f32(raw).shape == (3, 5, 7)


# -- container -------------------------------------------------------------


def test_round_trip_f32(tmp_path: Path):
    a = np.arange(12, dtype="<f4").reshape(3, 4)
    b = np.array([1.5, -2.5], dtype="<f4")
    path = write_safetensors(tmp_path / "m.safetensors", {"a": a, "b": b}, {"format": "pt"})

    with SafeTensors(path) as st:
        assert set(st.names) == {"a", "b"}
        assert st.metadata == {"format": "pt"}
        assert st.info("a").shape == (3, 4)
        np.testing.assert_array_equal(st.raw("a"), a)
        np.testing.assert_array_equal(st.f32("b"), b)
        assert st.total_params == 14
        assert st.total_bytes == 14 * 4


def test_bf16_tensor_through_the_container(tmp_path: Path):
    values = np.array([1.0, -2.0, 0.5, 3.140625], dtype=np.float32)
    path = write_safetensors(tmp_path / "m.safetensors", {"w": f32_to_bf16(values)})

    with SafeTensors(path) as st:
        assert st.info("w").dtype == "BF16"
        assert st.info("w").is_raw_bits
        assert st.raw("w").dtype == np.dtype("<u2")   # bits, not floats
        np.testing.assert_array_equal(st.f32("w"), values)


def test_raw_view_is_zero_copy_and_read_only(tmp_path: Path):
    a = np.arange(8, dtype="<f4")
    path = write_safetensors(tmp_path / "m.safetensors", {"a": a})
    with SafeTensors(path) as st:
        view = st.raw("a")
        assert not view.flags.writeable
        with pytest.raises(ValueError):
            view[0] = 1.0


def test_offsets_are_relative_to_the_data_buffer(tmp_path: Path):
    """The classic first bug: treating data_offsets as absolute file offsets.

    The second tensor starts at data offset 48, which is only correct if the
    reader adds the 8-byte prefix plus the header length.
    """
    a = np.arange(12, dtype="<f4")
    b = np.array([99.0], dtype="<f4")
    path = write_safetensors(tmp_path / "m.safetensors", {"a": a, "b": b})
    with SafeTensors(path) as st:
        assert st.info("b").begin == 48
        assert st.data_start == 8 + st.header_bytes
        np.testing.assert_array_equal(st.f32("b"), b)


def test_unknown_tensor_name(tmp_path: Path):
    path = write_safetensors(tmp_path / "m.safetensors", {"a": np.zeros(2, dtype="<f4")})
    with SafeTensors(path) as st:
        with pytest.raises(KeyError):
            st.info("nope")


# -- malformed files -------------------------------------------------------


def test_rejects_file_shorter_than_prefix(tmp_path: Path):
    p = tmp_path / "short.safetensors"
    p.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="shorter than"):
        SafeTensors(p)


def test_rejects_truncated_header(tmp_path: Path):
    p = tmp_path / "trunc.safetensors"
    p.write_bytes(struct.pack("<Q", 4096) + b'{"a":')
    with pytest.raises(ValueError, match="truncated"):
        SafeTensors(p)


def test_rejects_absurd_header_length(tmp_path: Path):
    p = tmp_path / "huge.safetensors"
    p.write_bytes(struct.pack("<Q", 2**40) + b"{}")
    with pytest.raises(ValueError, match="refusing"):
        SafeTensors(p)


def test_rejects_non_json_header(tmp_path: Path):
    body = b"not json at all"
    p = tmp_path / "bad.safetensors"
    p.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(ValueError, match="not valid JSON"):
        SafeTensors(p)


def test_rejects_unknown_dtype(tmp_path: Path):
    header = json.dumps({"a": {"dtype": "F128", "shape": [1], "data_offsets": [0, 16]}}).encode()
    p = tmp_path / "dt.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + bytes(16))
    with pytest.raises(ValueError, match="unsupported dtype"):
        SafeTensors(p)


def test_rejects_shape_that_disagrees_with_offsets(tmp_path: Path):
    """A shape needing 16 bytes but only 8 reserved means a corrupt file."""
    header = json.dumps({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}).encode()
    p = tmp_path / "mismatch.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
    with pytest.raises(ValueError, match="needs 16 bytes"):
        SafeTensors(p)


def test_rejects_offsets_past_the_end(tmp_path: Path):
    header = json.dumps({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
    p = tmp_path / "oob.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))
    with pytest.raises(ValueError, match="outside"):
        SafeTensors(p)


# -- the real model --------------------------------------------------------


@pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(),
    reason="model not downloaded; run tools/download_model.py",
)
def test_real_model_manifest_matches_config():
    from nanoinfer.config import ModelConfig
    from tools.inspect_weights import expected_tensors

    cfg = ModelConfig.from_model_dir(MODEL_DIR)
    expected = expected_tensors(cfg)

    with SafeTensors(MODEL_DIR / "model.safetensors") as st:
        assert set(st.names) == set(expected), "weight file does not match the config manifest"
        for name, shape in expected.items():
            assert st.info(name).shape == shape, name
            assert st.info(name).dtype == "BF16", name
        assert st.total_params == 494_032_768
