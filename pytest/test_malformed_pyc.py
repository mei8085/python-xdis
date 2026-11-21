"""
Malformed, truncated, and resource-abusive bytecode must fail safely
and consistently across all of the loading entry points:

* xdis.load.load_module                    -> ImportError
* xdis.load.load_module_from_file_object   -> ImportError
* xdis.unmarshal.load_code                 -> UnmarshalError (a ValueError)

No entry point may return a partial or placeholder object as though it
were valid code, and a failed load must not pollute the caller's
``code_objects`` dictionary.
"""

import io
import os
import tempfile
from struct import pack

import pytest

from xdis.load import load_module, load_module_from_file_object
from xdis.magics import by_version, magic2int
from xdis.unmarshal import UnmarshalError, load_code

# A complete CPython 2.7 code-object marshal stream, taken from stock
# CPython 2.7.18: compile('fn(*args)', '<fixture>', 'exec').
CODE27 = bytes.fromhex(
    "6300000000000000000200000040000000740e0000006500006501008c00000164000053"
    "28010000004e28020000007402000000666e74040000006172677328000000002800000000"
    "280000000073090000003c666978747572653e74080000003c6d6f64756c653e01000000"
    "7400000000"
)
MAGIC27 = by_version["2.7"]
MAGIC27_INT = magic2int(MAGIC27)
PYC27 = MAGIC27 + pack("<I", 0) + CODE27

MAGIC34_INT = magic2int(by_version["3.4"])


def load_module_path(data: bytes):
    """Run load_module() on ``data`` written to a temporary file."""
    fd, path = tempfile.mkstemp(suffix=".pyc")
    try:
        os.write(fd, data)
        os.close(fd)
        return load_module(path)
    finally:
        os.unlink(path)


def test_valid_pyc_loads_via_all_entry_points():
    version, _, _, co, *_ = load_module_from_file_object(io.BytesIO(PYC27))
    assert version == (2, 7)
    assert co.co_name == "<module>"

    version, _, _, co, *_ = load_module_path(PYC27)
    assert version == (2, 7)
    assert co.co_name == "<module>"

    co = load_code(CODE27, MAGIC27_INT, code_objects={})
    assert co.co_name == "<module>"


def test_short_files_fail_consistently():
    for data in (b"", b"\x03", MAGIC27[:2]):
        with pytest.raises(ImportError):
            load_module_from_file_object(io.BytesIO(data))
        with pytest.raises(ImportError):
            load_module_path(data)
    with pytest.raises(UnmarshalError):
        load_code(b"", MAGIC27_INT)


def test_unknown_magic_fails_consistently():
    bad_pyc = b"\xff\xff\r\n" + b"\0" * 100
    with pytest.raises(ImportError):
        load_module_from_file_object(io.BytesIO(bad_pyc))
    with pytest.raises(ImportError):
        load_module_path(bad_pyc)
    with pytest.raises(UnmarshalError):
        load_code(CODE27, 0xDEAD)


def test_truncation_fails_consistently_at_every_point():
    # Every proper prefix of a valid pyc is malformed; every entry
    # point must report an identifiable error rather than return a
    # partial object.
    for cut in range(len(PYC27)):
        prefix = PYC27[:cut]
        with pytest.raises(ImportError):
            load_module_from_file_object(io.BytesIO(prefix))
        if cut >= 8:
            with pytest.raises(UnmarshalError):
                load_code(prefix[8:], MAGIC27_INT)


def test_truncation_via_load_module_path():
    for cut in (0, 10, 40, len(PYC27) - 1):
        with pytest.raises(ImportError):
            load_module_path(PYC27[:cut])


def test_unknown_marshal_type_is_an_error_not_none(capsys):
    # Byte 0x01 is not a valid marshal type code.
    with pytest.raises(UnmarshalError):
        load_code(b"\x01", MAGIC27_INT)
    with pytest.raises(ImportError):
        load_module_from_file_object(io.BytesIO(MAGIC27 + pack("<I", 0) + b"\x01"))
    # Failures must be reported via the exception, not stderr noise.
    assert capsys.readouterr().err == ""


def test_bad_object_reference():
    # TYPE_REF to an object that was never interned.
    with pytest.raises(UnmarshalError):
        load_code(b"r" + pack("<I", 0), MAGIC34_INT)
    with pytest.raises(UnmarshalError):
        load_code(b"r" + pack("<I", 9999), MAGIC34_INT)


def test_bad_string_reference():
    # Python-2 TYPE_STRINGREF to a string that was never interned.
    with pytest.raises(UnmarshalError):
        load_code(b"R" + pack("<I", 0), MAGIC27_INT)


def test_deep_nesting_is_stopped():
    # Thousands of singly-nested tuples: must hit the depth limit with
    # an identifiable error, not a RecursionError or a crash.
    depth = 100_000
    payload = b"(" + pack("<I", 1)
    payload = payload * depth + b"N"
    with pytest.raises(UnmarshalError):
        load_code(payload, MAGIC27_INT)
    with pytest.raises(ImportError):
        load_module_from_file_object(
            io.BytesIO(MAGIC27 + pack("<I", 0) + payload)
        )


def test_huge_string_length_is_stopped():
    # A 4 GiB string length with almost no data behind it must fail
    # immediately instead of attempting a huge allocation/read.
    payload = b"s" + pack("<I", 0xFFFFFFFF) + b"tiny"
    with pytest.raises(UnmarshalError):
        load_code(payload, MAGIC27_INT)


def test_huge_long_is_stopped():
    # A long claiming ~2**30 digits (needs ~4 GB of digit data).
    payload = b"l" + pack("<i", 2**30)
    with pytest.raises(UnmarshalError):
        load_code(payload, MAGIC27_INT)


def test_failed_load_does_not_pollute_code_objects():
    # A tuple of (valid code object, garbage): the code object is fully
    # loaded before the failure, but the caller's dictionary must be
    # rolled back so partial results aren't mistaken for valid code.
    payload = b"(" + pack("<I", 2) + CODE27 + b"\x01"
    pyc = MAGIC27 + pack("<I", 0) + payload

    sentinel = object()
    code_objects = {"preexisting": sentinel}
    with pytest.raises(ImportError):
        load_module_from_file_object(io.BytesIO(pyc), code_objects=code_objects)
    assert code_objects == {"preexisting": sentinel}


def test_top_level_non_code_is_rejected():
    # A structurally valid marshal stream (NULL) that isn't a code
    # object must not be handed back as a successful load.
    pyc = MAGIC27 + pack("<I", 0) + b"0"
    with pytest.raises(ImportError):
        load_module_from_file_object(io.BytesIO(pyc))


def test_3531_magic_with_bad_disambiguation_byte():
    # Magic int 3531 (CPython 3.12 / RustPython 3.13) requires a
    # disambiguation byte at offset 0x10; anything unexpected is an
    # identifiable ImportError, not an AssertionError.
    pyc = pack("<Hcc", 3531, b"\r", b"\n") + b"\0" * 100
    with pytest.raises(ImportError):
        load_module_from_file_object(io.BytesIO(pyc))
