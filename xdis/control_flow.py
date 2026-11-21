#  Copyright (c) 2026 by Rocky Bernstein
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Version-correct control-flow graph construction for Python 3.6+
bytecode.

The point of this module is to be *one* place where the physical layout
of a ``co_code`` string is decided, so that jump targets, basic-block
nodes, predecessor/successor edges and error states cannot drift apart
across the encoding changes:

* before 3.6 instructions are 1 or 3 bytes; from 3.6 on every
  instruction is one fixed 2-byte word;
* from 3.10 on relative and absolute jump operands count 2-byte words
  instead of bytes;
* from 3.11 on ``CACHE`` words follow some instructions and are never
  instructions or legal entries;
* ``EXTENDED_ARG`` shifts the accumulated operand by 16 bits before 3.6
  and by 8 bits afterwards.

Consequences that this module makes explicit instead of leaving to each
caller:

* a jump may name an offset that is a legal instruction head, a
  continuation word of an ``EXTENDED_ARG`` chain, an operand byte inside
  a fixed-width word, a ``CACHE`` slot, or a position outside the code
  string. Only the first two are executable; the others are recorded as
  errors rather than being turned into bogus nodes;
* an exception-table entry contributes handler edges from every
  instruction it protects, but only when its target is executable and
  its range starts on an instruction boundary;
* graph nodes and edges are derived from the same decoded stream, so a
  data byte never becomes an instruction and no edge is silently lost
  because a version reported jumps in a different unit.
"""

from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from xdis.bytecode import get_logical_instruction_at_offset, parse_exception_table
from xdis.cross_dis import (
    falls_through,
    get_cache_entries,
    instruction_size,
)
from xdis.util import code2num


class EntryKind(Enum):
    """How an offset referenced by a jump or exception entry relates to the
    physical instruction layout."""

    # A canonical instruction head: the interpreter executes the same
    # instruction there as a linear sweep does.
    LEGAL = "legal"

    # A byte-aligned word that continues an EXTENDED_ARG chain. It is
    # executable (entering restarts argument accumulation) but the
    # instruction executed differs from the canonical sweep's, so it is
    # flagged rather than merged silently with the head.
    EXTENDED_ARG_CHAIN = "extended-arg-chain"

    # Not a physical instruction start: for fixed 2-byte wordcode this is
    # an odd offset, i.e. an operand byte.
    MID_INSTRUCTION = "mid-instruction"

    # The offset of an inline CACHE word (3.11+): data, never code.
    CACHE = "cache-slot"

    # Negative or at/past the end of the bytecode string.
    OUT_OF_BOUNDS = "out-of-bounds"

    @property
    def is_executable(self) -> bool:
        return self in (EntryKind.LEGAL, EntryKind.EXTENDED_ARG_CHAIN)


class Layout(NamedTuple):
    """Physical layout of a bytecode string.

    Fields:
      physical_starts:     offsets where a physical instruction word starts
      logical_starts:      physical starts that begin a logical instruction
                           (EXTENDED_ARG chain heads)
      chain_continuations: physical starts that continue an EXTENDED_ARG
                           chain (second and later words)
      cache_offsets:       offsets of inline CACHE words (3.11+)
    """

    physical_starts: Tuple[int, ...]
    logical_starts: Tuple[int, ...]
    chain_continuations: Tuple[int, ...]
    cache_offsets: Tuple[int, ...]


class TargetClass(NamedTuple):
    """Classification of an offset referenced by a jump or exception entry."""

    offset: int
    kind: EntryKind
    error: Optional[str]

    @property
    def is_executable(self) -> bool:
        return self.kind.is_executable


class FlowInstruction(NamedTuple):
    """A logical instruction node.

    ``offset`` is the head of the instruction, i.e. the first
    ``EXTENDED_ARG`` if the instruction has one. Operand-folding words and
    CACHE words are never nodes.
    """

    offset: int
    op: int
    opname: str
    arg: Optional[int]
    argval: Any
    has_extended_arg: bool


class FlowError(NamedTuple):
    """An inconsistent jump or exception-table reference.

    Fields:
      source:  offset of the jump/exception-protected instruction, or None
               for a malformed exception range
      target:  the offset the source names
      kind:    classification of the target
      opname:  jump opcode name, or None for an exception edge
      message: human readable explanation
    """

    source: Optional[int]
    target: int
    kind: EntryKind
    opname: Optional[str]
    message: str


class FlowGraph(NamedTuple):
    """Control-flow graph over logical instructions.

    Fields:
      opc:              opcode module the bytecode was decoded with
      bytecode:         the raw bytecode string
      layout:           physical layout used to build the graph
      instructions:     canonical logical instructions, in program order
      predecessors:     offset -> tuple of predecessor offsets. Keys are
                        canonical heads plus any EXTENDED_ARG-chain entries
                        named by a jump, so an executable entry is never
                        dropped.
      successors:       offset -> tuple of successor offsets (fallthrough,
                        jump, and exception edges).
      errors:           classification errors for non-executable entries
      exception_targets: offsets named by exception-table entries
      divergent_entries: offsets named by a jump that fall inside an
                        EXTENDED_ARG chain; executable but not canonical
                        instruction heads
    """

    opc: Any
    bytecode: Any
    layout: Layout
    instructions: Tuple[FlowInstruction, ...]
    predecessors: Dict[int, Tuple[int, ...]]
    successors: Dict[int, Tuple[int, ...]]
    errors: Tuple[FlowError, ...]
    exception_targets: Tuple[int, ...]
    divergent_entries: Tuple[int, ...]


def _version_tuple(opc) -> Tuple[int, ...]:
    """Best-effort version tuple for an opcode module.

    Graal opcode modules do not define ``python_version``; they carry the
    version on ``version_tuple`` only.
    """
    version = getattr(opc, "version_tuple", None)
    if version is not None:
        return tuple(version)
    python_version = getattr(opc, "python_version", None)
    return tuple(python_version) if python_version is not None else (0, 0)


def _is_graal(opc) -> bool:
    from xdis.version_info import PythonImplementation

    return getattr(opc, "python_implementation", None) == PythonImplementation.Graal


def bytecode_layout(bytecode, opc, instructions=None) -> Layout:
    """Return the physical layout of ``bytecode`` for ``opc``.

    Handles the fixed 2-byte wordcode of 3.6+ (skipping inline CACHE
    words) and the variable 1/3-byte encoding before 3.6.

    GraalVM uses a further variable 1-4 byte encoding whose widths are
    only known from its own disassembler. When ``instructions`` (its
    decoded stream) is supplied and ``opc`` is Graal, the layout is taken
    from those offsets instead of the generic sweep.
    """
    n = len(bytecode)

    if _is_graal(opc) and instructions is not None:
        logical = [inst.offset for inst in instructions]
        chain = [
            inst.offset for inst in instructions if inst.has_extended_arg
        ]
        return Layout(
            tuple(logical),
            tuple(o for o in logical if o not in set(chain)),
            tuple(chain),
            tuple(),
        )

    physical_starts: List[int] = []
    logical_starts: List[int] = []
    chain_continuations: List[int] = []
    cache_offsets: List[int] = []

    fixed_wordsize = _version_tuple(opc)[:2] >= (3, 6)
    extended_arg_op = getattr(opc, "EXTENDED_ARG", None)
    prev_was_extended_arg = False
    offset = 0
    while offset < n:
        op = code2num(bytecode, offset)
        physical_starts.append(offset)
        if prev_was_extended_arg:
            chain_continuations.append(offset)
        else:
            logical_starts.append(offset)
        prev_was_extended_arg = op == extended_arg_op
        if fixed_wordsize:
            caches = get_cache_entries(opc, opc.opname[op])
            for i in range(offset + 2, offset + 2 + 2 * caches, 2):
                if i < n:
                    cache_offsets.append(i)
            offset += 2 + 2 * caches
        else:
            offset += instruction_size(op, opc)
    return Layout(
        tuple(physical_starts),
        tuple(logical_starts),
        tuple(chain_continuations),
        tuple(cache_offsets),
    )


def classify_offset(offset: int, bytecode_len: int, layout: Layout) -> TargetClass:
    """Classify ``offset`` as a possible entry into the bytecode."""
    if offset < 0 or offset >= bytecode_len:
        return TargetClass(
            offset,
            EntryKind.OUT_OF_BOUNDS,
            f"offset {offset} is outside the {bytecode_len}-byte bytecode string",
        )
    if offset in layout.cache_offsets:
        return TargetClass(
            offset,
            EntryKind.CACHE,
            f"offset {offset} is an inline CACHE slot, not an instruction",
        )
    if offset in layout.chain_continuations:
        return TargetClass(
            offset,
            EntryKind.EXTENDED_ARG_CHAIN,
            None,
        )
    if offset in layout.logical_starts:
        return TargetClass(offset, EntryKind.LEGAL, None)
    return TargetClass(
        offset,
        EntryKind.MID_INSTRUCTION,
        f"offset {offset} is not an instruction boundary",
    )


def _coerce_bytecode(code) -> Tuple[Any, Any]:
    """Return ``(bytecode, code_object_or_None)`` for ``code``."""
    if hasattr(code, "co_code"):
        return code.co_code, code
    return code, None


def _exception_entries(code_object) -> Tuple[Any, ...]:
    if code_object is None:
        return tuple()
    entries = getattr(code_object, "exception_entries", None)
    if entries is not None:
        return tuple(entries)
    table = getattr(code_object, "co_exceptiontable", None)
    if table:
        return tuple(parse_exception_table(table))
    return tuple()


def _decode_instructions(
    bytecode, opc, layout: Layout, code_object=None
) -> Tuple[FlowInstruction, ...]:
    """Decode one logical instruction per canonical head.

    ``get_logical_instruction_at_offset`` folds an EXTENDED_ARG chain into
    the instruction that follows it; only the final instruction of the
    chain is returned as the logical instruction, with the head offset.
    """
    from xdis.version_info import PythonImplementation

    if getattr(opc, "python_implementation", None) == PythonImplementation.Graal:
        return _decode_instructions_graal(opc, code_object)

    instructions: List[FlowInstruction] = []
    for head in layout.logical_starts:
        try:
            decoded = list(
                get_logical_instruction_at_offset(
                    bytecode,
                    head,
                    opc,
                    labels=[],
                )
            )
        except IndexError:
            # Truncated trailing instruction: stop rather than read past
            # the end of the string.
            break
        if not decoded:
            continue
        final = decoded[-1]
        instructions.append(
            FlowInstruction(
                offset=head,
                op=final.opcode,
                opname=final.opname,
                arg=final.arg,
                argval=final.argval,
                has_extended_arg=len(decoded) > 1,
            )
        )
    return tuple(instructions)


def _decode_instructions_graal(opc, code_object) -> Tuple[FlowInstruction, ...]:
    """Decode GraalVM bytecode with its own variable-width reader.

    GraalVM instructions are 1-4 bytes and its disassembly, including
    jump-target resolution, lives in :mod:`xdis.bytecode_graal`; the
    generic wordcode decoder cannot describe it. This path requires the
    code object because GraalVM also records source-column maps.
    """
    if code_object is None:
        return tuple()
    from xdis.bytecode_graal import get_instructions_bytes_graal

    try:
        decoded = list(get_instructions_bytes_graal(code_object, opc))
    except Exception:
        return tuple()
    return tuple(
        FlowInstruction(
            offset=inst.offset,
            op=inst.opcode,
            opname=inst.opname,
            arg=inst.arg,
            argval=inst.argval,
            has_extended_arg=bool(inst.has_extended_arg),
        )
        for inst in decoded
    )


def _add_edge(
    successors: Dict[int, List[int]], predecessors: Dict[int, List[int]], src: int, dst: int
) -> None:
    succ = successors.setdefault(src, [])
    if dst not in succ:
        succ.append(dst)
    pred = predecessors.setdefault(dst, [])
    if src not in pred:
        pred.append(src)


def build_flow_graph(code, opc, bytecode=None) -> FlowGraph:
    """Build a control-flow graph for ``code`` (a code object or a raw
    bytecode string) under opcode module ``opc``.

    Jump targets are resolved with the version's operand unit and its
    inline CACHE layout; exception-table entries add handler edges. A
    target that is data (an operand byte or CACHE slot) or out of bounds
    is reported in ``errors`` and never becomes a node. A jump into an
    ``EXTENDED_ARG`` chain is kept as an executable node and recorded in
    ``divergent_entries``.
    """
    if bytecode is None:
        bytecode, code_object = _coerce_bytecode(code)
    else:
        _, code_object = _coerce_bytecode(code)

    if _is_graal(opc):
        # GraalVM widths come from its own disassembler, so decode first
        # and derive the layout from the decoded stream.
        graal_instructions = _decode_instructions_graal(opc, code_object)
        layout = bytecode_layout(bytecode, opc, instructions=graal_instructions)
        instructions = graal_instructions
    else:
        layout = bytecode_layout(bytecode, opc)
        instructions = _decode_instructions(
            bytecode, opc, layout, code_object=code_object
        )
    heads = {inst.offset for inst in instructions}

    successors: Dict[int, List[int]] = {}
    predecessors: Dict[int, List[int]] = {}
    errors: List[FlowError] = []
    divergent: List[int] = []

    def add_target(source: int, target: int, opname: Optional[str]) -> bool:
        target_class = classify_offset(target, len(bytecode), layout)
        if target_class.kind is EntryKind.LEGAL:
            if target in heads:
                return True
            # A legal head that the linear sweep did not reach because the
            # code string is truncated; still not a data byte.
            errors.append(
                FlowError(
                    source,
                    target,
                    target_class.kind,
                    opname,
                    f"target {target} is past the decoded instruction stream",
                )
            )
            return False
        if target_class.kind is EntryKind.EXTENDED_ARG_CHAIN:
            if target not in divergent:
                divergent.append(target)
            # Keep the node so an executable edge is not lost, but leave
            # its successors empty: decoding from here diverges from the
            # canonical sweep.
            successors.setdefault(target, [])
            return True
        errors.append(
            FlowError(
                source, target, target_class.kind, opname, target_class.error or ""
            )
        )
        return False

    # Fallthrough, jump and out-of-bounds edges.
    for index, inst in enumerate(instructions):
        next_offset = (
            instructions[index + 1].offset if index + 1 < len(instructions) else None
        )
        is_jump = inst.op in getattr(opc, "JREL_OPS", ()) or inst.op in getattr(
            opc, "JABS_OPS", ()
        )
        if is_jump and inst.argval is not None:
            if add_target(inst.offset, inst.argval, inst.opname):
                _add_edge(successors, predecessors, inst.offset, inst.argval)
        if falls_through(inst.op, opc) and next_offset is not None:
            _add_edge(successors, predecessors, inst.offset, next_offset)

    # Exception edges. An entry protects [start, end); every instruction
    # head in that range can transfer to the handler target.
    exception_targets: List[int] = []
    for entry in _exception_entries(code_object):
        start, end, target = entry.start, entry.end, entry.target
        if target not in exception_targets:
            exception_targets.append(target)
        start_class = classify_offset(start, len(bytecode), layout)
        if not start_class.is_executable:
            errors.append(
                FlowError(
                    None,
                    start,
                    start_class.kind,
                    None,
                    f"exception range start {start}: "
                    f"{start_class.error or 'not executable'}",
                )
            )
            continue
        if not 0 <= end <= len(bytecode):
            errors.append(
                FlowError(
                    None,
                    end,
                    EntryKind.OUT_OF_BOUNDS,
                    None,
                    f"exception range end {end} is outside the "
                    f"{len(bytecode)}-byte bytecode string",
                )
            )
            continue
        target_ok = add_target(start, target, None)
        for inst in instructions:
            if start <= inst.offset < end:
                if target_ok:
                    _add_edge(successors, predecessors, inst.offset, target)

    # Stable ordering of predecessor/successor lists by program order.
    order = {inst.offset: i for i, inst in enumerate(instructions)}
    all_offsets = set(successors) | set(predecessors)
    sorted_successors: Dict[int, Tuple[int, ...]] = {}
    sorted_predecessors: Dict[int, Tuple[int, ...]] = {}
    for offset in all_offsets:
        sorted_successors[offset] = tuple(sorted(set(successors.get(offset, ()))))
        sorted_predecessors[offset] = tuple(
            sorted(
                set(predecessors.get(offset, ())),
                key=lambda o: (order.get(o, len(order)), o),
            )
        )

    return FlowGraph(
        opc=opc,
        bytecode=bytecode,
        layout=layout,
        instructions=instructions,
        predecessors=sorted_predecessors,
        successors=sorted_successors,
        errors=tuple(errors),
        exception_targets=tuple(exception_targets),
        divergent_entries=tuple(sorted(divergent)),
    )
