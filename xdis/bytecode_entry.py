#  Copyright (c) 2025 by Rocky Bernstein
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

"""Multi-entry bytecode decoding for anti-analysis / obfuscated code.

Some obfuscators craft bytecode in which jumps land in the *middle* of an
instruction or in the middle of an ``EXTENDED_ARG`` chain. A naive linear
sweep then silently reinterprets operand bytes as harmless-looking
instructions, hiding the code that actually runs.

This module provides a decoding path that, for each entry point (offset 0,
jump targets, exception-table targets, or user-supplied offsets):

* classifies the entry as a legal instruction start, a byte-aligned
  position inside an ``EXTENDED_ARG`` chain, a misaligned (mid-instruction)
  position, or an out-of-bounds offset;
* decodes the instruction stream that the CPython interpreter would
  actually execute from that entry, using the entry's *own* initial
  ``EXTENDED_ARG`` state (none), rather than the state inherited from the
  linear sweep;
* flags every instruction whose offset is not an instruction start of the
  canonical linear sweep, so mid-instruction data is never silently
  presented as just another safe instruction;
* records where (if ever) a stream that started misaligned
  re-synchronizes with the canonical instruction boundaries.

Version-dependent encoding facts (see ``encoding_rules``):

* Before CPython 3.6 instructions are 1 byte (no operand) or 3 bytes
  (operand); from 3.6 on, every instruction is a fixed 2-byte word.
* ``EXTENDED_ARG`` shifts the accumulated operand by 16 bits per use
  before 3.6, and by 8 bits per use from 3.6 on.
* From 3.10 on, jump operands count 2-byte words rather than bytes.
* From 3.11 on, ``CACHE`` slots follow some instructions; these are
  physical words that are never legal jump entries.
* Entry recovery differs: with the fixed 2-byte word size (3.6+), an
  odd-offset entry can never rejoin the canonical boundaries (parity is
  preserved), while with the variable 1/3-byte encoding (pre 3.6) a
  misaligned stream may re-synchronize.
"""

from enum import Enum
from typing import List, NamedTuple, Optional, Set, Tuple

from xdis.bytecode import (
    get_logical_instruction_at_offset,
    is_fixed_wordsize_bytecode,
    next_offset,
)
from xdis.cross_dis import instruction_size
from xdis.instruction import Instruction
from xdis.util import code2num
from xdis.version_info import PythonImplementation


class EntryKind(Enum):
    """Classification of a bytecode entry offset."""

    # Aligned with a logical instruction start of the canonical linear
    # sweep from offset 0.
    LEGAL = "legal"

    # Byte-aligned with a physical instruction, but that instruction is
    # the continuation of an EXTENDED_ARG chain in the canonical linear
    # sweep. Jumping here restarts extended-argument accumulation, so the
    # executed argument differs from the linear-sweep interpretation.
    CHAIN_CONTINUATION = "chain-continuation"

    # Not aligned with any physical instruction boundary: the entry lands
    # on operand bytes (or, for 3.6+ wordcode, on an odd offset).
    OVERLAPPING = "overlapping"

    # Negative or at/past the end of the bytecode string.
    OUT_OF_BOUNDS = "out-of-bounds"


class EntryDecodeError(Exception):
    """Raised when bytecode cannot be decoded from a given entry point."""

    pass


class EncodingRules(NamedTuple):
    """Version-dependent instruction encoding facts used in entry decode.

    Fields:
      version_tuple:      the Python version these rules describe
      fixed_wordsize:     True if every instruction is one 2-byte word (3.6+)
      no_arg_size:        size in bytes of an instruction without an operand
      arg_size:           size in bytes of an instruction with an operand
      extended_arg_shift: bits each EXTENDED_ARG shifts the operand
                          (16 before 3.6, 8 from 3.6 on)
      has_cache_slots:    True if CACHE words can follow instructions (3.11+)
      jump_offset_unit:   unit that jump operands count in: bytes (1) before
                          3.10, 2-byte words (2) from 3.10 on
      resync_possible:    True if a stream entered mid-instruction can land
                          back on a canonical instruction boundary. This is
                          possible with the variable 1/3-byte encoding but
                          not with fixed 2-byte wordcode, where parity is
                          preserved.
    """

    version_tuple: tuple
    fixed_wordsize: bool
    no_arg_size: int
    arg_size: int
    extended_arg_shift: int
    has_cache_slots: bool
    jump_offset_unit: int
    resync_possible: bool


def encoding_rules(opc) -> EncodingRules:
    """Return the instruction-encoding rules for opcode module ``opc``."""
    version = opc.version_tuple
    fixed = is_fixed_wordsize_bytecode(opc)
    return EncodingRules(
        version_tuple=version,
        fixed_wordsize=fixed,
        no_arg_size=2 if fixed else 1,
        arg_size=2 if fixed else 3,
        extended_arg_shift=getattr(opc, "EXTENDED_ARG_SHIFT", 0),
        has_cache_slots="CACHE" in opc.opmap,
        jump_offset_unit=2 if version >= (3, 10) else 1,
        resync_possible=not fixed,
    )


def version_encoding_table(versions: Optional[List[tuple]] = None) -> List[EncodingRules]:
    """Compare instruction width, EXTENDED_ARG encoding, and entry-recovery
    behavior across CPython versions.

    ``versions`` is a list of version tuples, e.g. ``[(2, 7), (3, 6)]``.
    If omitted, a representative set of CPython versions spanning the
    encoding changes is used.
    """
    from xdis.op_imports import get_opcode_module

    if versions is None:
        versions = [
            (1, 5),
            (2, 7),
            (3, 0),
            (3, 5),
            (3, 6),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
        ]
    table = []
    for version in versions:
        opc = get_opcode_module(version, PythonImplementation.CPython)
        table.append(encoding_rules(opc))
    return table


class LinearScan(NamedTuple):
    """Result of the canonical linear sweep of a bytecode string.

    physical_starts: offsets where a physical instruction begins
    chain_continuations: physical starts that continue an EXTENDED_ARG
        chain (the second and later words of a logical instruction)
    truncated_at: offset of a final instruction whose operand bytes run
        past the end of the string, or None
    """

    physical_starts: Tuple[int, ...]
    chain_continuations: Tuple[int, ...]
    truncated_at: Optional[int]

    @property
    def logical_starts(self) -> Set[int]:
        """Offsets that begin a logical instruction (chain heads)."""
        return set(self.physical_starts) - set(self.chain_continuations)


def linear_scan(bytecode, opc) -> LinearScan:
    """Perform the canonical linear sweep of ``bytecode`` from offset 0.

    This is tolerant of truncated final instructions: instead of raising,
    the offending offset is recorded in ``truncated_at``.
    """
    n = len(bytecode)
    extended_arg_op = getattr(opc, "EXTENDED_ARG", None)
    physical_starts = []
    chain_continuations = []
    truncated_at = None
    offset = 0
    prev_was_extended_arg = False
    while offset < n:
        op = code2num(bytecode, offset)
        physical_starts.append(offset)
        if prev_was_extended_arg:
            chain_continuations.append(offset)
        size = instruction_size(op, opc)
        if offset + size > n:
            truncated_at = offset
            break
        prev_was_extended_arg = op == extended_arg_op
        offset += size
    return LinearScan(
        tuple(physical_starts), tuple(chain_continuations), truncated_at
    )


def classify_entry(
    bytecode, offset: int, opc, scan: Optional[LinearScan] = None
) -> EntryKind:
    """Classify ``offset`` as an entry point into ``bytecode``.

    Returns one of EntryKind.LEGAL, EntryKind.CHAIN_CONTINUATION,
    EntryKind.OVERLAPPING, or EntryKind.OUT_OF_BOUNDS.
    """
    n = len(bytecode)
    if offset < 0 or offset >= n:
        return EntryKind.OUT_OF_BOUNDS
    if scan is None:
        scan = linear_scan(bytecode, opc)
    if offset in scan.chain_continuations:
        return EntryKind.CHAIN_CONTINUATION
    if offset in scan.physical_starts:
        return EntryKind.LEGAL
    return EntryKind.OVERLAPPING


class EntryInstruction(NamedTuple):
    """An Instruction decoded while walking from some entry point, with
    annotations describing how trustworthy its position is.

    Fields:
      instruction:    the decoded Instruction
      entry_offset:   the entry offset this stream was decoded from
      entry_kind:     classification of that entry
      mid_instruction: True if this instruction's offset is not a logical
                      instruction start of the canonical linear sweep —
                      i.e. a linear sweep would have treated these bytes as
                      operand data or as part of an EXTENDED_ARG chain.
                      Such instructions are reported, never silently
                      treated as safe.
      resynchronized: True if this instruction's offset *is* a logical
                      start of the canonical linear sweep. Once True it
                      stays True for the rest of the stream.
    """

    instruction: Instruction
    entry_offset: int
    entry_kind: EntryKind
    mid_instruction: bool
    resynchronized: bool


def _entry_labels(bytecode, opc, exception_entries=None) -> List[int]:
    """Jump-target labels plus exception-table targets.

    Tolerant of truncated bytecode: if the label scan runs off the end of
    the string, fall back to just the exception-table targets rather than
    failing the whole decode.
    """
    try:
        labels = opc.findlabels(bytecode, opc)
    except IndexError:
        labels = []
    if exception_entries is not None:
        for _start, _end, target, _depth, _lasti in exception_entries:
            if target not in labels:
                labels.append(target)
    return labels


def decode_entry(
    bytecode,
    entry_offset: int,
    opc,
    varnames=None,
    names=None,
    constants=None,
    cells=None,
    linestarts=None,
    line_offset: int = 0,
    exception_entries=None,
    labels=None,
    scan: Optional[LinearScan] = None,
) -> List[EntryInstruction]:
    """Decode the instruction stream that CPython would execute if control
    entered ``bytecode`` at ``entry_offset``.

    Unlike a linear sweep, decoding starts with a fresh EXTENDED_ARG
    accumulation state, which is exactly what the interpreter does when a
    jump lands here.

    Every yielded EntryInstruction is annotated so that bytes which the
    canonical linear sweep regards as operand data are flagged
    (``mid_instruction``) rather than silently surfaced as safe
    instructions.

    Raises EntryDecodeError if ``entry_offset`` is out of bounds or the
    stream runs off the end of the bytecode.
    """
    kind = classify_entry(bytecode, entry_offset, opc, scan)
    if kind is EntryKind.OUT_OF_BOUNDS:
        raise EntryDecodeError(
            f"entry offset {entry_offset} is out of bounds for "
            f"{len(bytecode)}-byte bytecode string"
        )
    if scan is None:
        scan = linear_scan(bytecode, opc)
    if labels is None:
        labels = _entry_labels(bytecode, opc, exception_entries)

    logical_starts = scan.logical_starts
    n = len(bytecode)
    result: List[EntryInstruction] = []
    offset = entry_offset
    # A legal entry follows the canonical sweep from the start, so none of
    # its instructions (including EXTENDED_ARG chain members) are suspect.
    # A chain-continuation or overlapping entry is suspect until the
    # stream lands on a logical instruction start of the canonical sweep.
    in_sync = kind is EntryKind.LEGAL
    while offset < n:
        try:
            instructions = list(
                get_logical_instruction_at_offset(
                    bytecode,
                    offset,
                    opc,
                    varnames=varnames,
                    names=names,
                    constants=constants,
                    cells=cells,
                    linestarts=linestarts,
                    line_offset=line_offset,
                    exception_entries=exception_entries,
                    labels=labels,
                )
            )
        except IndexError:
            raise EntryDecodeError(
                f"bytecode entered at {entry_offset} is truncated at "
                f"offset {offset}"
            )
        if not instructions:
            break
        if not in_sync and instructions[0].offset in logical_starts:
            in_sync = True
        for inst in instructions:
            result.append(
                EntryInstruction(
                    instruction=inst,
                    entry_offset=entry_offset,
                    entry_kind=kind,
                    mid_instruction=not in_sync,
                    resynchronized=in_sync,
                )
            )
        last = instructions[-1]
        offset = next_offset(last.opcode, opc, last.offset)
    return result


class EntryDecodeResult(NamedTuple):
    """The outcome of decoding one entry point of a code object.

    Fields:
      entry_offset:  the entry offset decoded
      kind:          its EntryKind classification
      instructions:  tuple of EntryInstruction decoded from the entry
      resync_offset: first offset at which the stream rejoined the
                     canonical linear-sweep boundaries, or None if it
                     never did (or was never misaligned)
      error:         error message if decoding failed, else None
    """

    entry_offset: int
    kind: EntryKind
    instructions: Tuple[EntryInstruction, ...]
    resync_offset: Optional[int]
    error: Optional[str]


def find_entry_offsets(code_object, opc) -> List[int]:
    """Return the plausible entry offsets of a code object: offset 0, all
    jump targets, and all exception-table targets (3.11+)."""
    bytecode = code_object.co_code if hasattr(code_object, "co_code") else code_object
    offsets = {0}
    offsets.update(opc.findlabels(bytecode, opc))
    exception_entries = getattr(code_object, "exception_entries", None)
    if exception_entries:
        for _start, _end, target, _depth, _lasti in exception_entries:
            offsets.add(target)
    return sorted(offsets)


def _code_context(code_object, opc) -> dict:
    """Pull the operand-resolution tables out of a code object."""
    cellvars = (
        code_object.co_cellvars if hasattr(code_object, "co_cellvars") else tuple()
    )
    freevars = (
        code_object.co_freevars if hasattr(code_object, "co_freevars") else tuple()
    )
    linestarts = None
    if hasattr(opc, "findlinestarts"):
        linestarts = dict(opc.findlinestarts(code_object, dup_lines=True))
    return dict(
        varnames=code_object.co_varnames,
        names=code_object.co_names,
        constants=code_object.co_consts,
        cells=cellvars + freevars,
        linestarts=linestarts,
        exception_entries=getattr(code_object, "exception_entries", None),
    )


def decode_entries(
    code, opc, entries: Optional[List[int]] = None, strict: bool = False
) -> List[EntryDecodeResult]:
    """Decode ``code`` (a code object or a bytecode string) from each of
    its entry points and classify the results.

    ``entries`` defaults to ``find_entry_offsets``: offset 0 plus all jump
    and exception-handler targets. Entries that land mid-instruction or
    mid-EXTENDED_ARG-chain — the hallmark of jump-obfuscated bytecode —
    are decoded as the interpreter would execute them, but every
    instruction decoded from such data is flagged via
    ``EntryInstruction.mid_instruction``; it is never silently reported
    as an ordinary safe instruction.

    If ``strict`` is true, EntryDecodeError is raised on the first entry
    that is out of bounds or hits truncated bytecode; otherwise the error
    is recorded in the result's ``error`` field.
    """
    if hasattr(code, "co_code"):
        context = _code_context(code, opc)
        bytecode = code.co_code
    else:
        bytecode = code
        context = dict(
            varnames=None,
            names=None,
            constants=None,
            cells=None,
            linestarts=None,
            exception_entries=None,
        )

    exception_entries = context["exception_entries"]
    labels = _entry_labels(bytecode, opc, exception_entries)
    scan = linear_scan(bytecode, opc)

    if entries is None:
        entries = [0] + sorted(label for label in labels if label != 0)

    results: List[EntryDecodeResult] = []
    for entry in entries:
        kind = classify_entry(bytecode, entry, opc, scan)
        instructions: List[EntryInstruction] = []
        resync_offset: Optional[int] = None
        error: Optional[str] = None
        if kind is EntryKind.OUT_OF_BOUNDS:
            error = (
                f"entry offset {entry} is out of bounds for "
                f"{len(bytecode)}-byte bytecode string"
            )
            if strict:
                raise EntryDecodeError(error)
        else:
            try:
                instructions = decode_entry(
                    bytecode,
                    entry,
                    opc,
                    varnames=context["varnames"],
                    names=context["names"],
                    constants=context["constants"],
                    cells=context["cells"],
                    linestarts=context["linestarts"],
                    exception_entries=exception_entries,
                    labels=labels,
                    scan=scan,
                )
            except EntryDecodeError as err:
                error = str(err)
                if strict:
                    raise
            for entry_inst in instructions:
                if entry_inst.resynchronized:
                    resync_offset = entry_inst.instruction.offset
                    break
            if kind is EntryKind.LEGAL:
                # A legal entry is on a canonical boundary from the start;
                # there is nothing to recover from.
                resync_offset = None
        results.append(
            EntryDecodeResult(entry, kind, tuple(instructions), resync_offset, error)
        )
    return results
