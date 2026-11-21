# (C) Copyright 2020-2021, 2023-2025 by Rocky Bernstein
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

# Here, we are more closely modeling Python's ``dis`` module organization.
# However, it appears that Python's names and code have been copied a bit heavily from
# earlier versions of xdis (and without attribution).

from types import CodeType
from typing import List, Optional, Tuple

from xdis.util import (
    COMPILER_FLAG_NAMES,
    PYPY_COMPILER_FLAG_NAMES,
    better_repr,
    code2num,
)
from xdis.version_info import IS_GRAAL, PYTHON_IMPLEMENTATION, PythonImplementation


def _try_compile(source: str, name: str) -> CodeType:
    """Attempts to compile the given source, first as an expression and
    then as a statement if the first approach fails.

    Utility function to accept strings in functions that otherwise
    expect code objects
    """
    try:
        c = compile(source, name, "eval")
    except SyntaxError:
        c = compile(source, name, "exec")
    return c


def code_info(
    x, version_tuple: Tuple[int, ...], python_implementation: PythonImplementation
) -> str:
    """Formatted details of methods, functions, or code."""
    return format_code_info(
        get_code_object(x), version_tuple, python_implementation=python_implementation
    )


def get_code_object(x):
    """Helper to handle methods, functions, generators, strings and raw code objects"""
    if hasattr(x, "__func__"):  # Method
        x = x.__func__
    if hasattr(x, "__code__"):  # Function
        x = x.__code__
    elif hasattr(x, "func_code"):  # Function pre 2.7
        x = x.__code__
    elif hasattr(x, "gi_code"):  # Generator
        x = x.gi_code
    elif hasattr(x, "ag_code"):  # ...an asynchronous generator object, or
        x = x.ag_code
    elif hasattr(x, "cr_code"):  # ...a coroutine.
        x = x.cr_code
    # Handle source code.
    if isinstance(x, str):
        x = _try_compile(x, "<disassembly>")
    # By now, if we don't have a code object, we can't disassemble x.
    if hasattr(x, "co_code"):
        return x
    raise TypeError("don't know how to disassemble %s objects" % type(x).__name__)


# Number of inline CACHE words that follow an instruction, keyed by the
# Python version that first used that layout. These mirror CPython's
# ``_cache_format`` in ``Lib/opcode.py``. The counts are physical 2-byte
# words, not bytes.
#
# Layout changes over time:
#   3.11: first release with inline caches (LOAD_GLOBAL has 5 words)
#   3.12: LOAD_GLOBAL trimmed to 4, LOAD_ATTR grew to 9; FOR_ITER/SEND
#         gained a cache word
#   3.13: CONTAINS_OP, JUMP_BACKWARD, TO_BOOL and the conditional
#         POP_JUMP_IF_* ops gained caches
#   3.14: BINARY_OP grew to 5 words; CALL_KW added
#   3.15: GET_ITER, CALL_FUNCTION_EX and RESUME gained caches
#
# A name is looked up exactly first. A name carrying a specialization
# suffix (e.g. ``FOR_ITER_LIST`` in 3.14+) is mapped through
# ``_CACHE_SPECIALIZATION_BASES`` to its base opcode so a specialized
# instruction keeps the base's cache layout. The mapping is explicit
# rather than a prefix match so that an unrelated opcode which merely
# shares a prefix (``CALL_FUNCTION_EX`` versus ``CALL``) is not given the
# wrong cache count.
_CACHE_ENTRIES_311 = {
    "LOAD_GLOBAL": 5,
    "BINARY_OP": 1,
    "UNPACK_SEQUENCE": 1,
    "COMPARE_OP": 2,
    "BINARY_SUBSCR": 4,
    "LOAD_ATTR": 4,
    "STORE_ATTR": 4,
    "LOAD_METHOD": 10,
    "CALL": 4,
    "PRECALL": 1,
    "STORE_SUBSCR": 1,
}

_CACHE_ENTRIES_312 = {
    "LOAD_GLOBAL": 4,
    "BINARY_OP": 1,
    "UNPACK_SEQUENCE": 1,
    "COMPARE_OP": 1,
    "BINARY_SUBSCR": 1,
    "FOR_ITER": 1,
    "LOAD_SUPER_ATTR": 1,
    "LOAD_ATTR": 9,
    "STORE_ATTR": 4,
    "CALL": 3,
    "STORE_SUBSCR": 1,
    "SEND": 1,
}

_CACHE_ENTRIES_313 = {
    "LOAD_GLOBAL": 4,
    "BINARY_OP": 1,
    "UNPACK_SEQUENCE": 1,
    "COMPARE_OP": 1,
    "CONTAINS_OP": 1,
    "BINARY_SUBSCR": 1,
    "FOR_ITER": 1,
    "LOAD_SUPER_ATTR": 1,
    "LOAD_ATTR": 9,
    "STORE_ATTR": 4,
    "CALL": 3,
    "STORE_SUBSCR": 1,
    "SEND": 1,
    "JUMP_BACKWARD": 1,
    "TO_BOOL": 3,
    "POP_JUMP_IF_TRUE": 1,
    "POP_JUMP_IF_FALSE": 1,
    "POP_JUMP_IF_NONE": 1,
    "POP_JUMP_IF_NOT_NONE": 1,
}

_CACHE_ENTRIES_314 = {
    "LOAD_GLOBAL": 4,
    "BINARY_OP": 5,
    "UNPACK_SEQUENCE": 1,
    "COMPARE_OP": 1,
    "CONTAINS_OP": 1,
    "FOR_ITER": 1,
    "LOAD_SUPER_ATTR": 1,
    "LOAD_ATTR": 9,
    "STORE_ATTR": 4,
    "CALL": 3,
    "CALL_KW": 3,
    "STORE_SUBSCR": 1,
    "SEND": 1,
    "JUMP_BACKWARD": 1,
    "TO_BOOL": 3,
    "POP_JUMP_IF_TRUE": 1,
    "POP_JUMP_IF_FALSE": 1,
    "POP_JUMP_IF_NONE": 1,
    "POP_JUMP_IF_NOT_NONE": 1,
}

_CACHE_ENTRIES_315 = {
    "LOAD_GLOBAL": 4,
    "BINARY_OP": 5,
    "UNPACK_SEQUENCE": 1,
    "COMPARE_OP": 1,
    "CONTAINS_OP": 1,
    "FOR_ITER": 1,
    "GET_ITER": 1,
    "LOAD_SUPER_ATTR": 1,
    "LOAD_ATTR": 9,
    "STORE_ATTR": 4,
    "CALL": 3,
    "CALL_KW": 3,
    "CALL_FUNCTION_EX": 1,
    "STORE_SUBSCR": 1,
    "SEND": 1,
    "JUMP_BACKWARD": 1,
    "TO_BOOL": 3,
    "POP_JUMP_IF_TRUE": 1,
    "POP_JUMP_IF_FALSE": 1,
    "POP_JUMP_IF_NONE": 1,
    "POP_JUMP_IF_NOT_NONE": 1,
    "RESUME": 1,
}


# Instruction-name -> family name for specialized instructions that can
# appear in a ``co_code`` stream (for example after adaptive
# specialization, or in bytecode produced by a runtime that leaves
# specialized opcodes behind). ``get_cache_entries`` uses this to give a
# specialized opcode its base instruction's cache layout.
#
# The mapping mirrors CPython's ``_specializations`` in
# ``Lib/_opcode_metadata.py``. ``FOR_ITER_*`` and the ``BINARY_OP_*`` /
# ``TO_BOOL_*`` families are common enough to be worth naming even though
# a well-formed ``co_code`` normally holds only the base forms.
_CACHE_SPECIALIZATION_BASES = {
    # 3.11 - 3.13
    "LOAD_CONST_MORTAL": "LOAD_CONST",
    "LOAD_CONST_IMMORTAL": "LOAD_CONST",
    "CALL_FUNCTION_EX_INTRINSIC_1": "CALL_INTRINSIC_1",
    "FOR_ITER_LIST": "FOR_ITER",
    "FOR_ITER_TUPLE": "FOR_ITER",
    "FOR_ITER_RANGE": "FOR_ITER",
    "FOR_ITER_GEN": "FOR_ITER",
    "FOR_ITER_VIRTUAL": "FOR_ITER",
    "SEND_GEN": "SEND",
    "SEND_ASYNC_GEN": "SEND",
    "SEND_VIRTUAL": "SEND",
    "LOAD_GLOBAL_MODULE": "LOAD_GLOBAL",
    "LOAD_GLOBAL_BUILTIN": "LOAD_GLOBAL",
    "LOAD_SUPER_ATTR_ATTR": "LOAD_SUPER_ATTR",
    "LOAD_SUPER_ATTR_METHOD": "LOAD_SUPER_ATTR",
    "STORE_ATTR_INSTANCE_VALUE": "STORE_ATTR",
    "STORE_ATTR_SLOT": "STORE_ATTR",
    "STORE_ATTR_WITH_HINT": "STORE_ATTR",
    "STORE_SUBSCR_DICT": "STORE_SUBSCR",
    "STORE_SUBSCR_LIST_INT": "STORE_SUBSCR",
    "UNPACK_SEQUENCE_LIST": "UNPACK_SEQUENCE",
    "UNPACK_SEQUENCE_TUPLE": "UNPACK_SEQUENCE",
    "UNPACK_SEQUENCE_TWO_TUPLE": "UNPACK_SEQUENCE",
    "RESUME_CHECK": "RESUME",
    "RESUME_CHECK_JIT": "RESUME",
    "JUMP_BACKWARD_JIT": "JUMP_BACKWARD",
    "JUMP_BACKWARD_NO_JIT": "JUMP_BACKWARD",
}


def _add_cache_specializations() -> None:
    """Extend :data:`_CACHE_SPECIALIZATION_BASES` with the many
    ``LOAD_ATTR_*``, ``BINARY_OP_*``, ``COMPARE_OP_*``, ``CONTAINS_OP_*``,
    ``TO_BOOL_*`` and ``CALL_*`` specialized names.

    They are generated here rather than spelled out so the set stays
    short and easy to audit against CPython's ``_specializations``.
    """
    families = {
        "BINARY_OP": (
            "ADD_FLOAT", "ADD_INT", "ADD_UNICODE", "EXTEND", "MULTIPLY_FLOAT",
            "MULTIPLY_INT", "SUBTRACT_FLOAT", "SUBTRACT_INT", "INPLACE_ADD_UNICODE",
            "SUBSCR_DICT", "SUBSCR_GETITEM", "SUBSCR_LIST_INT",
            "SUBSCR_LIST_SLICE", "SUBSCR_STR_INT", "SUBSCR_TUPLE_INT",
            "SUBSCR_USTR_INT",
        ),
        "LOAD_ATTR": (
            "INSTANCE_VALUE", "SLOT", "WITH_HINT", "CLASS", "CLASS_WITH_METACLASS_CHECK",
            "GETATTRIBUTE_OVERRIDDEN", "METHOD_LAZY_DICT", "METHOD_NO_DICT",
            "METHOD_WITH_VALUES", "MODULE", "NONDESCRIPTOR_NO_DICT",
            "NONDESCRIPTOR_WITH_VALUES", "PROPERTY",
        ),
        "COMPARE_OP": ("FLOAT", "INT", "STR"),
        "CONTAINS_OP": ("DICT", "SET"),
        "TO_BOOL": ("ALWAYS_TRUE", "BOOL", "INT", "LIST", "NONE", "STR"),
        "UNPACK_SEQUENCE": ("LIST", "TUPLE", "TWO_TUPLE"),
    }
    for base, suffixes in families.items():
        for suffix in suffixes:
            _CACHE_SPECIALIZATION_BASES[f"{base}_{suffix}"] = base


_add_cache_specializations()


# Opcodes after which execution does not fall through to the next
# instruction in program order. ``opc.nofollow`` used to carry this, but it
# is incomplete from 3.12 (``RETURN_CONST`` is missing) and empty from 3.13
# on. Listing the names here keeps the "falls through" answer stable across
# every version and keeps a return from acquiring a bogus successor.
_NO_FALLTHROUGH_NAMES = frozenset(
    {
        # returns, raises, interpreter exits
        "STOP_CODE",
        "BREAK_LOOP",
        "CONTINUE_LOOP",
        "RETURN_VALUE",
        "RETURN_CONST",
        "RAISE_VARARGS",
        "RERAISE",
        "INTERPRETER_EXIT",
        # unconditional jumps across the versions
        "JUMP_ABSOLUTE",
        "JUMP_FORWARD",
        "JUMP_BACKWARD",
        "JUMP_BACKWARD_NO_INTERRUPT",
        "JUMP",
        "JUMP_NO_INTERRUPT",
        "JUMP_BACKWARD_JIT",
        "JUMP_BACKWARD_NO_JIT",
        "INSTRUMENTED_JUMP_FORWARD",
        "INSTRUMENTED_JUMP_BACKWARD",
    }
)


def falls_through(op: int, opc) -> bool:
    """Return True if instruction ``op`` can fall through to the next
    instruction in program order.

    Conditional jumps, ``FOR_ITER``, ``SEND`` and ``END_ASYNC_FOR`` all
    fall through; unconditional jumps and returns/raises do not. This is
    authoritative across the versions that dropped or never completed
    ``opc.nofollow``.
    """
    return opc.opname[op] not in _NO_FALLTHROUGH_NAMES


def get_cache_entries(opc, opname: str) -> int:
    """Return the number of inline ``CACHE`` words that follow ``opname``
    in an instruction list for opcode module ``opc``.

    CPython 3.11 is the first release with inline caches. Before that, no
    instruction is followed by cache words.

    The lookup is by exact name, matching CPython's ``_cache_format``.
    A name carrying a specialization suffix (e.g. ``FOR_ITER_LIST``,
    ``SEND_GEN``, ``LOAD_ATTR_SLOT``, ``JUMP_BACKWARD_JIT``) is mapped to
    its base opcode first. That base is only accepted when the suffix is
    a known specialization of that base, so an unrelated opcode that
    merely shares a prefix — ``CALL_FUNCTION_EX`` versus ``CALL`` — is
    not given the wrong cache count.
    """
    version = tuple(opc.version_tuple[:2])
    if version < (3, 11):
        return 0
    # Inline caches are a CPython 3.11+ code layout. Some other
    # implementations share the version number (PyPy 3.11, RustPython
    # 3.12/3.13) but have no CACHE opcode; they must not inherit the
    # layout or every following instruction would be wrongly skipped.
    if "CACHE" not in getattr(opc, "opmap", {}):
        return 0
    if version >= (3, 15):
        table = _CACHE_ENTRIES_315
    elif version >= (3, 14):
        table = _CACHE_ENTRIES_314
    elif version >= (3, 13):
        table = _CACHE_ENTRIES_313
    elif version >= (3, 12):
        table = _CACHE_ENTRIES_312
    elif version >= (3, 11):
        table = _CACHE_ENTRIES_311
    else:
        return 0

    if opname in table:
        return table[opname]

    base = _CACHE_SPECIALIZATION_BASES.get(opname)
    if base is not None:
        return table.get(base, 0)
    return 0


def get_cache_size_313(opname: str) -> int:
    """Number of inline CACHE words following ``opname`` under the
    Python 3.13 layout.

    Kept for compatibility with callers that predate the
    version-parameterized :func:`get_cache_entries`. New code should use
    :func:`get_cache_entries` with the opcode module being decoded.
    """
    return _CACHE_ENTRIES_313.get(opname, 0)


# For compatibility
_get_cache_size_313 = get_cache_size_313


def _jump_unit(opc) -> int:
    """Bytes per unit of a relative jump's operand.

    From Python 3.10 on, jump operands are counted in 2-byte code units
    ("words"); before that they are counted in bytes. Absolute jumps use
    the same unit.
    """
    return 2 if tuple(opc.version_tuple[:2]) >= (3, 10) else 1


def _is_backward_jump(opc, opname: str) -> bool:
    """Return True if ``opname`` jumps backwards.

    From 3.11 relative jumps encode direction in the opcode name. From
    3.14, ``END_ASYNC_FOR`` also carries a backwards target (it is not a
    control-flow jump, but its operand points back at the ``END_SEND``).
    """
    version = tuple(opc.version_tuple[:2])
    if "JUMP_BACKWARD" in opname:
        return True
    if version >= (3, 14) and "END_ASYNC_FOR" in opname:
        return True
    return False


def get_jump_target(offset: int, op: int, arg: int, opc) -> Optional[int]:
    """Return the byte offset that ``arg`` of jump instruction ``op`` at
    ``offset`` targets, or None if ``op`` is not a jump.

    The computation is version aware:

    * relative operands count bytes before 3.10 and 2-byte words after;
    * the base offset is the instruction that *follows* the jump, which
      is the jump word plus any inline ``CACHE`` words (3.11+) it owns,
      so a target does not accidentally land on cache data;
    * 3.11+ encodes direction in the opcode name (``JUMP_BACKWARD``);
    * 3.14+ gives ``END_ASYNC_FOR`` a backwards target.
    """
    opname = opc.opname[op]
    unit = _jump_unit(opc)
    if op in opc.JREL_OPS:
        delta = arg * unit
        # The base is the instruction that follows the jump: the jump
        # itself (1 or 3 bytes before 3.6, one 2-byte word after) plus
        # any inline CACHE words (3.11+) that belong to the jump.
        base = offset + instruction_size(op, opc) + 2 * get_cache_entries(opc, opname)
        target = base - delta if _is_backward_jump(opc, opname) else base + delta
        return target
    if op in opc.JABS_OPS:
        return arg * unit
    return None


def findlabels(code: bytes, opc):
    if opc.version_tuple < (3, 6) or IS_GRAAL:
        return findlabels_pre_310(code, opc)

    return findlabels_310(code, opc)


def findlabels_310(code: bytes, opc):
    """Returns a list of instruction offsets in the supplied bytecode
    which are the targets of some sort of jump instruction.
    """
    labels = []
    for offset, op, arg in unpack_opargs_bytecode_310(code, opc):
        if arg is None:
            continue
        label = get_jump_target(offset, op, arg, opc)
        if label is None:
            continue
        if label not in labels:
            labels.append(label)
    return labels


def findlabels_pre_310(code, opc):
    """Returns a list of instruction offsets in the supplied bytecode
    which are the targets of some sort of jump instruction.

    This handles the variable-width (1 or 3 byte) instructions used
    before 3.6, where jump operands are counted in bytes. GraalVM bytecode
    shares the variable-width layout.
    """
    offsets = []
    for offset, op, arg in unpack_opargs_bytecode(code, opc):
        if arg is None:
            continue
        jump_offset = -1
        if op in opc.JREL_OPS:
            op_len = op_size(op, opc)
            jump_offset = offset + op_len + arg
        elif op in opc.JABS_OPS:
            jump_offset = arg
        if jump_offset >= 0:
            if jump_offset not in offsets:
                offsets.append(jump_offset)
    return offsets


# For the `co_lines` attribute, we want to emit the full form, omitting
# the (350, 360, No line number) and empty entries.

NO_LINE_NUMBER = -128


def findlinestarts(code, dup_lines: bool = False):
    """Find the offsets in a byte code which are start of lines in the source.

    Generate pairs (offset, lineno) as described in Python/compile.c.
    """

    if hasattr(code, "co_lines") and hasattr(code, "co_linetable"):
        # Taken from 3.10 findlinestarts
        lastline = None
        for start, _, line in code.co_lines():
            if line is not None and line != lastline:
                lastline = line
                yield start, line

    else:
        lineno_table = code.co_lnotab

        if isinstance(lineno_table, dict):
            # We have an uncompressed line-number table
            # The below could be done with a Python generator, but
            # we want to be Python 2.x compatible.
            for addr, lineno in lineno_table.items():
                yield addr, lineno
            # For 3.8 we have to fall through to the return rather
            # than add raise StopIteration
        elif len(lineno_table) == 0:
            yield 0, code.co_firstlineno
        else:
            if isinstance(lineno_table[0], int):
                byte_increments = list(code.co_lnotab[0::2])
                line_deltas = list(code.co_lnotab[1::2])
            else:
                byte_increments = [ord(c) for c in code.co_lnotab[0::2]]
                line_deltas = [ord(c) for c in code.co_lnotab[1::2]]
            bytecode_len = len(code.co_code)

            lastlineno = None
            lineno = code.co_firstlineno
            offset = 0
            byte_incr = 0
            for byte_incr, line_delta in zip(byte_increments, line_deltas):
                if byte_incr:
                    if lineno != lastlineno or dup_lines and 0 < byte_incr < 255:
                        yield offset, lineno
                        lastlineno = lineno
                        pass
                    if offset >= bytecode_len:
                        # The rest of the ``lnotab byte offsets are past the end of
                        # the bytecode; any line numbers for these have been removed.
                        return
                    offset += byte_incr
                    pass
                if line_delta >= 0x80:
                    # line_deltas is an array of 8-bit *signed* integers
                    line_delta -= 0x100
                lineno += line_delta
            if lineno != lastlineno or (dup_lines and 0 < byte_incr < 255):
                yield offset, lineno

    return


def instruction_size(op, opc) -> int:
    """For a given opcode, `op`, in opcode module `opc`,
    return the size, in bytes, of an `op` instruction.

    This is the size of the opcode (one byte) and any operand it has.
    In Python before version 3.6, this will be either 1 or 3 bytes.
    In Python 3.6 or later, it is 2 bytes: a "word"."""
    if op < opc.HAVE_ARGUMENT:
        return 2 if opc.version_tuple >= (3, 6) else 1
    else:
        return 2 if opc.version_tuple >= (3, 6) else 3


# Compatibility
op_size = instruction_size


def show_code(
    co,
    version_tuple: Tuple[int, ...],
    file=None,
    python_implementation=PYTHON_IMPLEMENTATION,
) -> None:
    """Print details of methods, functions, or code to *file*.

    If *file* is not provided, the output is printed on stdout.
    """
    if file is None:
        print(code_info(co, version_tuple, python_implementation))
    else:
        file.write(code_info(co, version_tuple, python_implementation) + "\n")


def op_has_argument(opcode: int, opc) -> bool:
    """
    Return True if `opcode` instruction has an operand.

    From Python 3.12 CPython (and for RustPython) the opcode module ships
    an explicit ``hasarg`` table. That table, not ``HAVE_ARGUMENT``, is
    authoritative: from 3.13 on, ``HAVE_ARGUMENT`` is a low sentinel (44)
    while some opcodes below it still take an operand (e.g. the 3.13
    ``MAKE_FUNCTION`` operand moved to ``SET_FUNCTION_ATTRIBUTE``) and
    some opcodes at or above it do not (e.g. ``WITH_EXCEPT_START``).
    Using ``opcode >= HAVE_ARGUMENT`` there would decode operands that
    are not there and skip operands that are.
    """
    hasarg = getattr(opc, "hasarg", None)
    if hasarg is not None:
        return opcode in hasarg
    return opcode >= opc.HAVE_ARGUMENT


def pretty_flags(flags, python_implementation=PYTHON_IMPLEMENTATION) -> str:
    """Return pretty representation of code flags."""
    names = []
    result = "0x%08x" % flags
    for i in range(32):
        flag = 1 << i
        if flags & flag:
            if (
                python_implementation == PythonImplementation.PyPy
                and flag in PYPY_COMPILER_FLAG_NAMES
            ):
                names.append(PYPY_COMPILER_FLAG_NAMES.get(flag, hex(flag)))
            else:
                names.append(COMPILER_FLAG_NAMES.get(flag, hex(flag)))
            flags ^= flag
            if not flags:
                break
    else:
        names.append(hex(flags))
    names.reverse()
    return "%s (%s)" % (result, " | ".join(names))


def format_code_info(
    co,
    version_tuple: tuple,
    name=None,
    python_implementation=PYTHON_IMPLEMENTATION,
    file_offset: Optional[tuple] = None,
) -> str:
    if not name:
        name = co.co_name
    lines = []

    if not (name == "?" and version_tuple <= (2, 4)):
        lines.append("# Method Name:       %s" % name)

    # Python before version 2.4 and earlier didn't store a name for the main routine.
    # Later versions use "<module>"
    lines.append("# Filename:          %s" % co.co_filename)

    if file_offset:
        lines.append("# Offset in file:    0x%x" % file_offset[0])

    if version_tuple >= (1, 3):
        lines.append("# Argument count:    %s" % co.co_argcount)

    if version_tuple >= (3, 8) and hasattr(co, "co_posonlyargcount"):
        lines.append("# Position-only argument count: %s" % co.co_posonlyargcount)

    if version_tuple >= (3, 0) and hasattr(co, "co_kwonlyargcount"):
        lines.append("# Keyword-only arguments: %s" % co.co_kwonlyargcount)

    pos_argc = co.co_argcount
    if hasattr(co, "co_nlocals"):
        lines.append("# Number of locals:  %s" % co.co_nlocals)
    if version_tuple >= (1, 5):
        lines.append("# Stack size:        %s" % co.co_stacksize)
        pass

    if version_tuple >= (1, 3):
        lines.append(
            "# Flags:             %s" % pretty_flags(co.co_flags, python_implementation)
        )

    if version_tuple >= (1, 5):
        lines.append("# First Line:        %s" % co.co_firstlineno)
    # if co.co_freevars:
    #     lines.append("# Freevars:      %s" % str(co.co_freevars))
    if co.co_consts:
        lines.append("# Constants:")
        for i, c in enumerate(co.co_consts):
            lines.append("# %4d: %s" % (i, better_repr(c)))
    if co.co_names:
        lines.append("# Names:")
        for i_n in enumerate(co.co_names):
            lines.append("# %4d: %s" % i_n)
    if co.co_varnames:
        lines.append("# Varnames:")
        lines.append("#\t%s" % ", ".join(co.co_varnames))
        pass
    if pos_argc > 0:
        lines.append("# Positional arguments:")
        lines.append("#\t%s" % ", ".join(co.co_varnames[:pos_argc]))
        pass
    if len(co.co_varnames) > pos_argc:
        lines.append("# Local variables:")
        for i, n in enumerate(co.co_varnames[pos_argc:]):
            lines.append("# %4d: %s" % (pos_argc + i, n))
    if version_tuple > (2, 0):
        if co.co_freevars:
            lines.append("# Free variables:")
            for i_n in enumerate(co.co_freevars):
                lines.append("# %4d: %s" % i_n)
                pass
            pass
        if co.co_cellvars:
            lines.append("# Cell variables:")
            for i_n in enumerate(co.co_cellvars):
                lines.append("# %4d: %s" % i_n)
                pass
            pass

    if file_offset:
        lines.append("# co_code offset in file: 0x%x" % file_offset[1])

    return "\n".join(lines)


def format_exception_table(bytecode, version_tuple) -> str:
    if version_tuple < (3, 11) or not hasattr(bytecode, "exception_entries"):
        return ""
    lines: List[str] = ["ExceptionTable:"]
    for entry in bytecode.exception_entries:
        lasti = " lasti" if entry.lasti else ""
        end = entry.end - 2
        lines.append(
            f"  {entry.start} to {end} -> {entry.target} [{entry.depth}]{lasti}"
        )
    return "\n".join(lines)


def extended_arg_val(opc, val):
    return val << opc.EXTENDED_ARG_SHIFT


def unpack_opargs_bytecode_310(code, opc):
    """Yield ``(offset, opcode, arg)`` for fixed 2-byte wordcode.

    Inline ``CACHE`` words (3.11+) are skipped: they are never
    instructions and, were they yielded, they would both lose the
    fallthrough edge across them and let a jump target computed from the
    following instruction land on cache data.

    As CPython's ``dis._unpack_opargs`` does from 3.11, operands are
    interpreted as signed 32-bit values so that ``EXTENDED_ARG`` chains
    that overflow wrap to a negative operand instead of producing an
    enormous positive one.
    """
    extended_arg = 0
    try:
        n = len(code)
    except TypeError:
        code = code.co_code
        n = len(code)

    wrap_signed = tuple(opc.version_tuple[:2]) >= (3, 11)
    caches = 0
    for offset in range(0, n - 1, 2):
        if caches:
            caches -= 1
            continue
        op = code2num(code, offset)
        caches = get_cache_entries(opc, opc.opname[op])
        if op_has_argument(op, opc):
            arg = code2num(code, offset + 1) | extended_arg
            extended_arg = extended_arg_val(opc, arg) if op == opc.EXTENDED_ARG else 0
            # CPython stores the arg as a signed integer from 3.11 on;
            # EXTENDED_ARG chains beyond 31 bits overflow and wrap
            # negative.
            if wrap_signed and extended_arg >= _INT_OVERFLOW:
                extended_arg -= 2 * _INT_OVERFLOW
        else:
            arg = None
            extended_arg = 0
        yield offset, op, arg


# CPython's dis module treats jump/EXTENDED_ARG operands as signed 32-bit
# integers (see ``dis._INT_OVERFLOW``).
_INT_OVERFLOW = 1 << 31


# This is modified from Python 3.6's ``dis`` module
def unpack_opargs_bytecode(code, opc):
    extended_arg = 0
    try:
        n = len(code)
    except TypeError:
        code = code.co_code
        n = len(code)

    offset = 0
    while offset < n:
        prev_offset = offset
        op = code2num(code, offset)
        offset += 1
        if op_has_argument(op, opc):
            if offset + 1 >= n:
                # A final instruction whose operand runs past the end of
                # the bytecode string: stop rather than read garbage.
                break
            arg = code2num(code, offset) | code2num(code, offset + 1) << 8
            arg |= extended_arg
            extended_arg = (
                extended_arg_val(opc, arg)
                if hasattr(opc, "EXTENDED_ARG") and op == opc.EXTENDED_ARG
                else 0
            )
            offset += 2
        else:
            arg = None
        yield prev_offset, op, arg


def get_jump_target_maps(code, opc):
    """Returns a dictionary where the key is an offset and the values are
    a list of instruction offsets which can get run before that
    instruction. This includes jump instructions as well as non-jump
    instructions. Therefore, the keys of the dictionary are reachable
    instructions. The values of the dictionary may be useful in control-flow
    analysis.

    The walk is version aware:

    * instructions are decoded with the width and ``EXTENDED_ARG``
      semantics of ``opc`` (see :func:`unpack_opargs_bytecode_310` and
      :func:`unpack_opargs_bytecode`);
    * relative jump targets are computed in the version's operand unit
      through :func:`get_jump_target`, which also skips inline
      ``CACHE`` words;
    * jumps to an offset outside the bytecode string are reported as a
      predecessor entry too, so an out-of-bounds target is not silently
      dropped (it maps to a key no instruction occupies, making it easy
      to detect), and
    * ``CACHE`` words never contribute or receive an edge.

    ``prev`` values are instruction-head offsets in increasing program
    order. An ``EXTENDED_ARG`` chain is one logical instruction located at
    the offset of its first word, matching
    :func:`xdis.control_flow.build_flow_graph`. A jump into the middle of a
    chain is attributed to the physical offset the jump names.
    """
    is_wordcode = tuple(opc.version_tuple[:2]) >= (3, 6)
    unpack = unpack_opargs_bytecode_310 if is_wordcode else unpack_opargs_bytecode
    extended_arg_op = getattr(opc, "EXTENDED_ARG", None)

    offset2prev = {}
    prev_node = -1
    prev_next = -1
    chain_head = -1
    for offset, op, arg in unpack(code, opc):
        if op == extended_arg_op:
            if chain_head < 0:
                chain_head = offset
            continue
        node = chain_head if chain_head >= 0 else offset
        chain_head = -1
        # Fallthrough edge: the previous instruction flows to this one.
        # Inline CACHE words (3.11+) sit between them physically but are
        # not instructions, so the gap must be exactly the previous
        # instruction's cache words.
        if prev_node >= 0 and prev_next == node:
            prev_list = offset2prev.get(node, [])
            prev_list.append(prev_node)
            offset2prev[node] = prev_list
        if falls_through(op, opc):
            prev_node = node
        else:
            prev_node = -1
        prev_next = (
            offset
            + instruction_size(op, opc)
            + 2 * get_cache_entries(opc, opc.opname[op])
        )
        if arg is not None:
            jump_offset = get_jump_target(offset, op, arg, opc)
            if jump_offset is not None:
                prev_list = offset2prev.get(jump_offset, [])
                prev_list.append(node)
                offset2prev[jump_offset] = prev_list
    return offset2prev


# In CPython, this is C code. We redo this in Python using the
# information in opc.
def xstack_effect(opcode, opc, oparg: int = 0, jump=None):
    """Compute the stack effect of opcode with argument oparg, using
    oppush and oppop tables in opc.

    If the code has a jump target and jump is True, stack_effect()
    will return the stack effect of jumping. If jump is False, it will
    return the stack effect of not jumping. And if jump is None
    (default), it will return the maximal stack effect of both cases.
    """
    version_tuple = opc.version_tuple
    pop, push = opc.oppop[opcode], opc.oppush[opcode]
    opname = opc.opname[opcode]
    if version_tuple >= (3, 0):
        if opname in "BUILD_CONST_KEY_MAP" and version_tuple >= (3, 12):
            return -oparg
        if opname == "BUILD_MAP" and version_tuple >= (3, 5):
            return 1 - (2 * oparg)
        if opname in ("UNPACK_SEQUENCE",):
            return oparg - 1
        elif opname in ("UNPACK_EX"):
            return (oparg & 0xFF) + (oparg >> 8)
        elif opname == "BUILD_INTERPOLATION":
            # 3.14+ only
            return -2 if oparg & 1 else -1

    if opname in (
        "BUILD_LIST",
        "BUILD_SET",
        "BUILD_STRING",
        "BUILD_TUPLE",
    ) and version_tuple >= (3, 12):
        return 1 - oparg
    elif opname in ("BUILD_SLICE") and version_tuple < (3, 14):
        # this was 2.7 only, but 3.0 through 3.13 also pop the step in a[x:y:z]
        # (3.14 changed BUILD_SLICE and its opcode table already handles it)
        return -2 if oparg == 3 else -1
    elif opname == "FORMAT_VALUE" and version_tuple >= (3, 6):
        # f"{x:>10}" also pops the format spec; the table only counted the value
        return -1 if oparg & 0x04 else 0
    elif opname == "LOAD_ATTR" and version_tuple >= (3, 12):
        return 1 if oparg & 1 else 0
    elif opname == "MAKE_FUNCTION":
        if version_tuple >= (3, 5):
            # was 10, but a nested function with defaults, kwdefaults, and a closure is 11 (15 with annotations)
            if 0 <= oparg <= (10 if version_tuple == (3, 5) else 15):
                if version_tuple == (3, 5):
                    return [-1, -2, -3, -3, -2, -3, -3, -4, -2, -3, -3, -4][oparg]
                elif (3, 6) <= version_tuple < (3, 11):
                    return [-1, -2, -2, -3, -2, -3, -3, -4, -2, -3, -3, -4, -3, -4, -4, -5][oparg]
                else:
                    # only flags 0-2 were handled, so closures and annotations gave None;
                    # this is the 3.6-3.10 table plus one since 3.11 no longer pops the qualname
                    return [0, -1, -1, -2, -1, -2, -2, -3, -1, -2, -2, -3, -2, -3, -3, -4][oparg]
            else:
                return None
    elif opname in ("CALL", "INSTRUMENTED_CALL") and version_tuple >= (3, 12):
        return -oparg - 1
    elif opname in ("CALL_KW", "INSTRUMENTED_CALL_KW"):
        return -2 - oparg
    elif opname == "CALL_FUNCTION_EX":
        if version_tuple >= (3, 14):
            return -3
        if (3, 5) <= version_tuple < (3, 11):
            return -2 if oparg & 1 else -1
        elif 0 <= oparg <= 3:
            return -3 if oparg & 1 else -2
        else:
            return None
    elif opname in (
        "INSTRUMENTED_LOAD_SUPER_ATTR",
        "LOAD_SUPER_ATTR",
    ) and version_tuple >= (3, 12):
        if opname == "INSTRUMENTED_LOAD_SUPER_ATTR" and version_tuple >= (3, 14):
            return -2
        return -1 if oparg & 1 else -2
    elif opname == "LOAD_GLOBAL" and version_tuple >= (3, 11):
        return 2 if oparg & 1 else 1
    elif opname == "PRECALL" and version_tuple >= (3, 11):
        return -oparg
    elif opname == "RAISE_VARARGS" and version_tuple >= (3, 12):
        return -oparg
    if push >= 0 and pop >= 0:
        return push - pop
    elif pop < 0:
        # The amount popped depends on oparg, and opcode class
        if opcode in opc.VARGS_OPS:
            return push - oparg + (pop + 1)
        elif opcode in opc.NARGS_OPS:
            return -oparg + pop + push
    return -100


if __name__ == "__main__":
    from dis import findlabels as findlabels_std

    my_code = findlabels.__code__.co_code
    from xdis.op_imports import get_opcode_module

    my_opc = get_opcode_module()
    assert findlabels(my_code, my_opc) == findlabels_std(my_code)
