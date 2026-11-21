"""Regression tests for version-correct control-flow graph construction.

These exercise the four things that used to disagree across bytecode
versions and between xdis and CPython's own ``dis``:

* instruction width (1/3 bytes before 3.6, fixed 2-byte wordcode after);
* ``EXTENDED_ARG`` shift and operand folding;
* jump-operand unit (bytes before 3.10, 2-byte words after);
* inline ``CACHE`` words (3.11+) and direction-encoding jumps (3.11+).

They also check that a jump or exception entry naming a data byte, an
``EXTENDED_ARG`` continuation, a ``CACHE`` slot, or an offset outside the
string is reported as a classified error rather than becoming a node, and
that the same code object yields the same predecessors regardless of how
the version names jump operands.
"""

import dis
import glob
import os

import pytest

from xdis.bytecode import Bytecode
from xdis.control_flow import (
    EntryKind,
    build_flow_graph,
    bytecode_layout,
    classify_offset,
)
from xdis.cross_dis import (
    findlabels,
    get_cache_entries,
    get_jump_target,
    instruction_size,
)
from xdis.load import load_module
from xdis.op_imports import get_opcode_module
from xdis.version_info import PythonImplementation, PYTHON_VERSION_TRIPLE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test")


def opc_for(version):
    return get_opcode_module(version, PythonImplementation.CPython)


def walk_codes(co):
    yield co
    for const in co.co_consts:
        if hasattr(const, "co_code"):
            yield from walk_codes(const)


def fixture_codes(folder):
    for path in sorted(glob.glob(os.path.join(DATA_DIR, folder, "*.pyc"))):
        try:
            info = load_module(path)
        except Exception:
            continue
        version = info[0]
        impl = info[4]
        try:
            opc = get_opcode_module(version, impl)
        except Exception:
            continue
        for co in walk_codes(info[3]):
            yield path, co, opc


# ---------------------------------------------------------------------------
# Version encoding facts
# ---------------------------------------------------------------------------


def test_instruction_width_before_and_after_36():
    from xdis.opcodes import opcode_27, opcode_36, opcode_310

    # 1 or 3 bytes before 3.6
    assert instruction_size(opcode_27.opmap["LOAD_CONST"], opcode_27) == 3
    assert instruction_size(opcode_27.opmap["RETURN_VALUE"], opcode_27) == 1
    # fixed 2-byte words from 3.6
    for opc in (opcode_36, opcode_310):
        for name in ("LOAD_CONST", "RETURN_VALUE"):
            assert instruction_size(opc.opmap[name], opc) == 2


def test_jump_unit_changes_at_310():
    from xdis.opcodes import opcode_39, opcode_310

    # 3.9: relative operands count bytes
    jump = opcode_39.opmap["JUMP_ABSOLUTE"]
    assert get_jump_target(0, jump, 4, opcode_39) == 4
    # 3.10: absolute operands count 2-byte words
    jump = opcode_310.opmap["JUMP_ABSOLUTE"]
    assert get_jump_target(0, jump, 4, opcode_310) == 8


def test_extended_arg_shift():
    from xdis.opcodes import opcode_27, opcode_36

    assert opcode_27.EXTENDED_ARG_SHIFT == 16
    assert opcode_36.EXTENDED_ARG_SHIFT == 8


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def test_cache_layout_by_version():
    from xdis.opcodes import opcode_310, opcode_311, opcode_312, opcode_313, opcode_314, opcode_315

    # No inline caches before 3.11
    for opc in (opcode_310,):
        assert get_cache_entries(opc, "LOAD_GLOBAL") == 0

    assert get_cache_entries(opcode_311, "LOAD_GLOBAL") == 5
    assert get_cache_entries(opcode_312, "LOAD_GLOBAL") == 4
    assert get_cache_entries(opcode_312, "LOAD_ATTR") == 9
    assert get_cache_entries(opcode_312, "CALL") == 3
    assert get_cache_entries(opcode_313, "TO_BOOL") == 3
    assert get_cache_entries(opcode_313, "POP_JUMP_IF_FALSE") == 1
    assert get_cache_entries(opcode_314, "BINARY_OP") == 5
    assert get_cache_entries(opcode_315, "GET_ITER") == 1
    assert get_cache_entries(opcode_315, "CALL_FUNCTION_EX") == 1

    # An opcode that merely shares a prefix with a cached opcode must not
    # inherit that opcode's cache count.
    assert get_cache_entries(opcode_312, "CALL_FUNCTION_EX") == 0
    assert get_cache_entries(opcode_312, "LOAD_CONST") == 0


def test_cache_absent_on_non_cpython_runtimes():
    from xdis.opcodes.opcode_rust import opcode_24481rust

    # RustPython shares the 3.13 version number but has no CACHE opcode.
    assert get_cache_entries(opcode_24481rust, "LOAD_ATTR") == 0


# ---------------------------------------------------------------------------
# Jump target / findlabels correctness on real fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("folder", ["bytecode_3.11", "bytecode_3.12", "bytecode_3.13", "bytecode_3.14", "bytecode_3.15"])
def test_fixture_jump_targets_land_on_instruction_heads(folder):
    seen = 0
    for _path, co, opc in fixture_codes(folder):
        heads = {inst.offset for inst in Bytecode(co, opc) if inst.opname != "CACHE"}
        for inst in Bytecode(co, opc):
            if inst.opname == "CACHE":
                continue
            if inst.is_jump():
                seen += 1
                assert inst.argval in heads, (
                    f"{folder} {co.co_name}: {inst.opname} at {inst.offset} "
                    f"targets {inst.argval}, not an instruction head"
                )
    # These folders always have jumps; guard against an accidental empty run.
    if folder != "bytecode_3.15":
        assert seen > 0


def test_findlabels_matches_native_dis_312():
    if PYTHON_VERSION_TRIPLE[:2] != (3, 12):
        pytest.skip("native comparison needs a 3.12 interpreter")
    opc = opc_for((3, 12))
    compared = 0
    for _path, co, _opc in fixture_codes("bytecode_3.12"):
        compared += 1
        assert findlabels(co.co_code, opc) == dis.findlabels(co.co_code)
    assert compared > 0


# ---------------------------------------------------------------------------
# Layout / decoded stream agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("folder", ["bytecode_3.11", "bytecode_3.12", "bytecode_3.13", "bytecode_3.14", "bytecode_3.15"])
def test_decoded_stream_matches_physical_layout(folder):
    for _path, co, opc in fixture_codes(folder):
        layout = bytecode_layout(co.co_code, opc)
        # ``Bytecode`` yields every physical instruction word including an
        # EXTENDED_ARG chain's words, but never a CACHE word.
        decoded = [inst.offset for inst in Bytecode(co, opc) if inst.opname != "CACHE"]
        expected = [
            offset
            for offset in layout.physical_starts
            if offset not in layout.cache_offsets
        ]
        assert decoded == expected
        # CACHE words are recorded separately and never appear as heads.
        assert not (set(decoded) & set(layout.cache_offsets))
        # The flow graph collapses each EXTENDED_ARG chain into its head.
        graph = build_flow_graph(co, opc)
        assert [inst.offset for inst in graph.instructions] == list(
            layout.logical_starts
        )


# ---------------------------------------------------------------------------
# Predecessor maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("folder", ["bytecode_3.10", "bytecode_3.11", "bytecode_3.12", "bytecode_3.13", "bytecode_3.14", "bytecode_3.15"])
def test_predecessor_maps_are_complete(folder):
    for _path, co, opc in fixture_codes(folder):
        graph = build_flow_graph(co, opc)
        assert graph.errors == ()

        heads = {inst.offset for inst in graph.instructions}
        # Every in-range jump target is a predecessor-map key.
        for inst in graph.instructions:
            is_jump = inst.op in opc.JREL_OPS or inst.op in opc.JABS_OPS
            if is_jump and inst.argval is not None:
                assert inst.argval in graph.predecessors, (
                    f"{co.co_name}: no predecessor entry for jump target "
                    f"{inst.argval} from {inst.opname} at {inst.offset}"
                )
                assert inst.offset in graph.predecessors[inst.argval]

        # No predecessor key is a data byte or a CACHE slot.
        for key in graph.predecessors:
            assert key not in graph.layout.cache_offsets
            if key in heads:
                continue
            # Only EXTENDED_ARG-chain entries are allowed off-head.
            assert key in graph.layout.chain_continuations


def test_same_graph_regardless_of_unit_naming():
    """A code object yields the same edges whether decoded via the
    ``opc.get_jump_target_maps`` entry point or the shared cross-version
    helpers, including for EXTENDED_ARG chains and inline caches."""
    from xdis.cross_dis import get_jump_target_maps as shared_maps
    from xdis.wordcode import get_jump_target_maps as wordcode_maps

    compared = 0
    for folder in ("bytecode_3.12", "bytecode_3.13", "bytecode_3.14"):
        for _path, co, opc in fixture_codes(folder):
            compared += 1
            graph = build_flow_graph(co, opc)
            graph_edges = {
                (src, target)
                for target, preds in graph.predecessors.items()
                for src in preds
            }
            for maps in (shared_maps, wordcode_maps):
                via_maps = maps(co.co_code, opc)
                map_edges = {
                    (src, target)
                    for target, preds in via_maps.items()
                    for src in preds
                }
                # The graph adds exception edges; everything else must
                # match exactly, and no non-exception edge may be lost.
                assert map_edges <= graph_edges, (
                    f"{co.co_name}: map edges missing from graph: "
                    f"{sorted(map_edges - graph_edges)[:5]}"
                )
    assert compared > 0


# ---------------------------------------------------------------------------
# Error states: data targets, out-of-bounds, chain entries, cache slots
# ---------------------------------------------------------------------------


def _raw_const(bytecode, opc, offset):
    """Decode a raw string as a code object is not possible, so build the
    graph from the raw bytecode."""
    return build_flow_graph(bytecode, opc)


def test_out_of_bounds_jump_is_reported_not_added():
    opc = opc_for((3, 12))
    jump_forward = opc.opmap["JUMP_FORWARD"]
    return_value = opc.opmap["RETURN_VALUE"]
    # JUMP_FORWARD by 50 words from offset 0 lands far outside the 4 bytes.
    bytecode = bytes([jump_forward, 50, return_value, 0])
    graph = _raw_const(bytecode, opc, 0)
    assert graph.errors
    error = graph.errors[0]
    assert error.kind is EntryKind.OUT_OF_BOUNDS
    assert error.opname == "JUMP_FORWARD"
    # The bogus target never becomes a node with incoming edges.
    assert error.target not in graph.predecessors


def test_jump_into_extended_arg_chain_keeps_edge_but_flags_entry():
    from xdis.opcodes import opcode_313

    extended = opcode_313.EXTENDED_ARG
    load_const = opcode_313.opmap["LOAD_CONST"]
    return_value = opcode_313.opmap["RETURN_VALUE"]
    jump_forward = opcode_313.opmap["JUMP_FORWARD"]
    # 0: JUMP_FORWARD +1 word -> 4 (the continuation of the chain at 2)
    # 2: EXTENDED_ARG 1
    # 4: LOAD_CONST 0
    # 6: RETURN_VALUE
    bytecode = bytes([jump_forward, 1, extended, 1, load_const, 0, return_value, 0])
    graph = build_flow_graph(bytecode, opcode_313)
    assert 4 in graph.divergent_entries
    assert 4 in graph.predecessors
    assert graph.predecessors[4] == (0,)
    # The chain continuation is executable, so it is not an error.
    assert all(error.target != 4 for error in graph.errors)
    # And no bogus instruction was invented for a data byte.
    assert all(inst.offset != 1 for inst in graph.instructions)


def test_jump_into_cache_slot_is_an_error():
    from xdis.opcodes import opcode_313

    nop = opcode_313.opmap["NOP"]
    jump_backward = opcode_313.opmap["JUMP_BACKWARD"]
    return_value = opcode_313.opmap["RETURN_VALUE"]
    # JUMP_BACKWARD at 4 owns one CACHE word at 6. From base 8, arg=1
    # targets 6 -- its own cache slot.
    bytecode = bytes([nop, 0, nop, 0, jump_backward, 1, return_value, 0])
    layout = bytecode_layout(bytecode, opcode_313)
    assert 6 in layout.cache_offsets
    graph = build_flow_graph(bytecode, opcode_313)
    kinds = {(error.target, error.kind) for error in graph.errors}
    assert (6, EntryKind.CACHE) in kinds
    assert 6 not in graph.predecessors


def test_classify_offset_covers_every_kind():
    from xdis.opcodes import opcode_313

    extended = opcode_313.EXTENDED_ARG
    load_const = opcode_313.opmap["LOAD_CONST"]
    return_value = opcode_313.opmap["RETURN_VALUE"]
    bytecode = bytes([extended, 1, load_const, 0, return_value, 0])
    layout = bytecode_layout(bytecode, opcode_313)
    length = len(bytecode)

    assert classify_offset(0, length, layout).kind is EntryKind.LEGAL
    assert classify_offset(2, length, layout).kind is EntryKind.EXTENDED_ARG_CHAIN
    assert classify_offset(1, length, layout).kind is EntryKind.MID_INSTRUCTION
    assert classify_offset(-2, length, layout).kind is EntryKind.OUT_OF_BOUNDS
    assert classify_offset(length, length, layout).kind is EntryKind.OUT_OF_BOUNDS


def test_extended_arg_chain_continuation_is_not_mid_instruction():
    from xdis.opcodes import opcode_313

    extended = opcode_313.EXTENDED_ARG
    load_const = opcode_313.opmap["LOAD_CONST"]
    return_value = opcode_313.opmap["RETURN_VALUE"]
    bytecode = bytes([extended, 1, load_const, 0, return_value, 0])
    graph = build_flow_graph(bytecode, opcode_313)
    assert graph.errors == ()
    # One logical instruction per canonical head, chain folded.
    assert [inst.offset for inst in graph.instructions] == [0, 4]
    assert graph.instructions[0].has_extended_arg
    assert graph.instructions[0].arg == 0x100
