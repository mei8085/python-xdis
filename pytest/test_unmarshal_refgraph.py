"""Object reference graph integrity when unmarshalling cross-version pyc data.

The marshal format (version 3+, Python 3.4+) serializes a graph, not a
tree: objects tagged with FLAG_REF occupy numbered reference slots and a
later TYPE_REF tag points back at a slot, so one object can be reached
through several paths and containers can be recursive.  These tests pin
the behaviour that shared objects are not copied, references are not lost
or mis-targeted, and recursive structures keep their cycles.

Streams are hand-built from the marshal wire format so that the tests do
not depend on the running interpreter's marshal version; they are decoded
with several historical magic numbers to prove the parsing path is
version-independent.
"""

import io
from struct import pack

import pytest

from xdis.magics import by_version, magic2int
from xdis.unmarshal import VersionIndependentUnmarshaller

FLAG_REF = 0x80
TYPE_REF = ord("r")
TYPE_SMALL_TUPLE = ord(")")
TYPE_TUPLE = ord("(")
TYPE_LIST = ord("[")
TYPE_DICT = ord("{")
TYPE_SET = ord("<")
TYPE_FROZENSET = ord(">")
TYPE_NULL = ord("0")
TYPE_INT = ord("i")
TYPE_SHORT_ASCII_INTERNED = ord("Z")


def unmarshal(data, version="3.12"):
    um = VersionIndependentUnmarshaller(
        io.BytesIO(data), magic2int(by_version[version]), False, code_objects={}
    )
    return um.load()


def interned(text):
    raw = text.encode("ascii")
    assert len(raw) < 256
    return bytes([TYPE_SHORT_ASCII_INTERNED | FLAG_REF, len(raw)]) + raw


def ref(slot):
    return bytes([TYPE_REF]) + pack("<i", slot)


def flagged_int(value):
    return bytes([TYPE_INT | FLAG_REF]) + pack("<i", value)


# Every marshal-version-3+ magic must decode the graph the same way.
VERSIONS = ("3.4", "3.7", "3.10", "3.12")


@pytest.mark.parametrize("version", VERSIONS)
def test_interned_string_back_reference_is_the_same_object(version):
    # slots: 0 = outer tuple, 1 = interned string, then ref(1)
    stream = (
        bytes([TYPE_SMALL_TUPLE | FLAG_REF, 2]) + interned("shared_value") + ref(1)
    )
    result = unmarshal(stream, version)
    assert result == ("shared_value", "shared_value")
    assert result[0] is result[1]


@pytest.mark.parametrize("version", VERSIONS)
def test_full_size_tuple_also_reserves_its_slot_first(version):
    # TYPE_TUPLE (not the small variant) must reserve slot 0 before its
    # children are read, exactly like marshal.c does.
    stream = (
        bytes([TYPE_TUPLE | FLAG_REF])
        + pack("<i", 2)
        + interned("shared_value")
        + ref(1)
    )
    result = unmarshal(stream, version)
    assert result[0] is result[1] == "shared_value"


@pytest.mark.parametrize("version", VERSIONS)
def test_shared_nested_tuple_is_not_duplicated(version):
    # slots: 0 = outer tuple, 1 = inner tuple, 2/3 = its ints.  ref(1)
    # must hand back the very inner tuple object rather than a rebuilt copy.
    inner = (
        bytes([TYPE_SMALL_TUPLE | FLAG_REF, 2]) + flagged_int(1) + flagged_int(2)
    )
    stream = bytes([TYPE_SMALL_TUPLE | FLAG_REF, 2]) + inner + ref(1)
    result = unmarshal(stream, version)
    assert result == ((1, 2), (1, 2))
    assert result[0] is result[1]
    assert isinstance(result[0], tuple)


@pytest.mark.parametrize("version", VERSIONS)
def test_self_referential_small_tuple_keeps_its_cycle(version):
    # tuple(slot 0) whose only element is ref(0): the tuple contains itself.
    stream = bytes([TYPE_SMALL_TUPLE | FLAG_REF, 1]) + ref(0)
    result = unmarshal(stream, version)
    assert isinstance(result, tuple)
    assert result[0] is result
    # repr() must terminate, proving the cycle is real rather than re-read.
    assert repr(result) == "((...),)"


@pytest.mark.parametrize("version", VERSIONS)
def test_self_referential_full_tuple_keeps_its_cycle(version):
    stream = bytes([TYPE_TUPLE | FLAG_REF]) + pack("<i", 1) + ref(0)
    result = unmarshal(stream, version)
    assert result[0] is result


@pytest.mark.parametrize("version", VERSIONS)
def test_nested_tuple_back_reference_resolves_mid_construction(version):
    # outer(slot0) = [ inner(slot1) = ( ref(1), interned "a"(slot2) ) ]
    stream = (
        bytes([TYPE_SMALL_TUPLE | FLAG_REF, 1])
        + bytes([TYPE_SMALL_TUPLE | FLAG_REF, 2])
        + ref(1)
        + interned("a")
    )
    result = unmarshal(stream, version)
    assert result[0][0] is result[0]
    assert result[0][1] == "a"


@pytest.mark.parametrize("version", VERSIONS)
def test_self_referential_list_keeps_its_cycle(version):
    stream = bytes([TYPE_LIST | FLAG_REF]) + pack("<i", 1) + ref(0)
    result = unmarshal(stream, version)
    assert result[0] is result


@pytest.mark.parametrize("version", VERSIONS)
def test_self_referential_dict_keeps_its_cycle(version):
    stream = (
        bytes([TYPE_DICT | FLAG_REF]) + interned("k") + ref(0) + bytes([TYPE_NULL])
    )
    result = unmarshal(stream, version)
    assert result["k"] is result


@pytest.mark.parametrize("version", VERSIONS)
def test_shared_set_back_reference_is_the_same_object(version):
    # slots: 0 = outer tuple, 1 = set, 2 = its element, ref(1) shares the set.
    set_body = bytes([TYPE_SET | FLAG_REF]) + pack("<i", 1) + flagged_int(8)
    stream = bytes([TYPE_SMALL_TUPLE | FLAG_REF, 2]) + set_body + ref(1)
    result = unmarshal(stream, version)
    assert result == ({8}, {8})
    assert result[0] is result[1]
    assert isinstance(result[0], set)


@pytest.mark.parametrize("version", VERSIONS)
def test_shared_frozenset_back_reference_is_the_same_object(version):
    frozen_body = (
        bytes([TYPE_FROZENSET | FLAG_REF]) + pack("<i", 1) + flagged_int(7)
    )
    stream = bytes([TYPE_SMALL_TUPLE | FLAG_REF, 2]) + frozen_body + ref(1)
    result = unmarshal(stream, version)
    assert result == (frozenset({7}), frozenset({7}))
    assert result[0] is result[1]


@pytest.mark.parametrize("version", VERSIONS)
def test_reference_slot_zero_is_a_valid_index(version):
    # The first FLAG_REF object is slot 0, so ref(0) must resolve to it.
    # Earlier code wrongly treated zero as "out of range".
    stream = bytes([TYPE_SMALL_TUPLE | FLAG_REF, 2]) + interned("s") + ref(0)
    result = unmarshal(stream, version)
    assert result[1] is result


@pytest.mark.parametrize("version", VERSIONS)
def test_out_of_range_reference_raises_instead_of_returning_wrong_object(version):
    stream = bytes([TYPE_REF]) + pack("<i", 99)
    with pytest.raises(ValueError):
        unmarshal(stream, version)


@pytest.mark.parametrize("version", VERSIONS)
def test_string_shared_across_nesting_levels(version):
    # outer(slot0) = [ inner(slot1) = (interned "deep"(slot2),) ], ref(2)
    inner = bytes([TYPE_SMALL_TUPLE | FLAG_REF, 1]) + interned("deep")
    stream = bytes([TYPE_SMALL_TUPLE | FLAG_REF, 2]) + inner + ref(2)
    result = unmarshal(stream, version)
    assert result == (("deep",), "deep")
    assert result[0][0] is result[1]


def test_recursive_tuples_are_real_tuples_not_build_placeholders():
    # On CPython the in-place tuple builder must hand back genuine tuples
    # at every level, including the recursive one.
    stream = bytes([TYPE_SMALL_TUPLE | FLAG_REF, 1]) + ref(0)
    result = unmarshal(stream)
    assert type(result) is tuple
    assert type(result[0]) is tuple


@pytest.mark.parametrize("version", VERSIONS)
def test_mutual_tuple_and_list_cycle(version):
    # a(slot0) = ( b(slot1), ) where b = [ ref(0) ] -> b[0] is a.
    stream = (
        bytes([TYPE_SMALL_TUPLE | FLAG_REF, 1])
        + bytes([TYPE_LIST | FLAG_REF])
        + pack("<i", 1)
        + ref(0)
    )
    result = unmarshal(stream, version)
    assert isinstance(result[0], list)
    assert result[0][0] is result


@pytest.mark.parametrize("version", VERSIONS)
def test_mutual_dict_and_list_cycle(version):
    # d(slot0) = {"k": L(slot1)} where L = [ ref(0) ].
    stream = (
        bytes([TYPE_DICT | FLAG_REF])
        + interned("k")
        + bytes([TYPE_LIST | FLAG_REF])
        + pack("<i", 1)
        + ref(0)
        + bytes([TYPE_NULL])
    )
    result = unmarshal(stream, version)
    assert result["k"][0] is result


@pytest.mark.parametrize("version", VERSIONS)
def test_mutual_tuple_tuple_cycle(version):
    # a(slot0) = ( b(slot1), ) where b = ( ref(0), ).
    stream = (
        bytes([TYPE_SMALL_TUPLE | FLAG_REF, 1])
        + bytes([TYPE_SMALL_TUPLE | FLAG_REF, 1])
        + ref(0)
    )
    result = unmarshal(stream, version)
    assert type(result) is tuple
    assert type(result[0]) is tuple
    assert result[0][0] is result
    # Cycle must print with the ellipsis marker rather than recurse forever.
    assert repr(result) == "(((...),),)"


def test_many_recursive_tuples_survive_garbage_collection():
    # Stress the CPython in-place tuple builder: build a large batch of
    # self-referential tuples and force GC, checking refcounts stay sound.
    import gc

    stream = bytes([TYPE_SMALL_TUPLE | FLAG_REF, 1]) + ref(0)
    roots = [unmarshal(stream) for _ in range(500)]
    for cycle in roots:
        assert cycle[0] is cycle
    del roots
    gc.collect()
    gc.collect()
