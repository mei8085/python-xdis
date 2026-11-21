# (C) Copyright 2018, 2020-2021, 2023, 2025 by Rocky Bernstein
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

"""Python disassembly functions specific to wordcode from Python 3.6+"""
from xdis.bytecode import op_has_argument
from xdis.cross_dis import (
    falls_through,
    get_cache_entries,
    get_jump_target,
    instruction_size,
    unpack_opargs_bytecode_310,
)


def unpack_opargs_wordcode(code, opc):
    """Yield ``(offset, opcode, arg)`` for fixed 2-byte wordcode.

    Inline ``CACHE`` words (3.11+) are skipped and operands are folded
    with the version's ``EXTENDED_ARG`` shift, exactly as CPython's
    ``dis._unpack_opargs`` does — otherwise the physical layout and the
    jump arithmetic derived from it drift apart for 3.11+ bytecode.
    """
    if isinstance(code, (bytes, bytearray, memoryview)):
        yield from unpack_opargs_bytecode_310(code, opc)
        return

    # The remaining path handles bytecode supplied as a string (for
    # example Python 3.x bytecode inspected from a 2.x interpreter).
    extended_arg = 0
    try:
        n = len(code)
    except TypeError:
        code = code.co_code
        n = len(code)

    caches = 0
    for i in range(0, n - 1, 2):
        if caches:
            caches -= 1
            continue
        op = ord(code[i])
        caches = get_cache_entries(opc, opc.opname[op])
        if op_has_argument(op, opc):
            arg = ord(code[i + 1]) | extended_arg
            extended_arg = (arg << 8) if op == opc.EXTENDED_ARG else 0
        else:
            arg = None
            extended_arg = 0
        yield i, op, arg


def findlabels(code, opc):
    """Returns a list of instruction offsets in the supplied bytecode
    which are the targets of jump instruction.
    """
    unpack_opargs = (
        unpack_opargs_wordcode
        if opc.version_tuple < (3, 10)
        else unpack_opargs_bytecode_310
    )

    offsets = []
    for offset, op, arg in unpack_opargs(code, opc):
        if arg is None:
            continue
        jump_offset = get_jump_target(offset, op, arg, opc)
        if jump_offset is None:
            continue
        if jump_offset not in offsets:
            offsets.append(jump_offset)
    return offsets


def get_jump_target_maps(code, opc) -> dict:
    """Returns a dictionary where the key is an offset, and the values are
    a list of instruction offsets which can get run before that
    instruction. This includes jump instructions as well as non-jump
    instructions. Therefore, the keys of the dictionary are reachable
    instructions. The values of the dictionary may be useful in control-flow
    analysis.

    This mirrors :func:`xdis.cross_dis.get_jump_target_maps` for 3.6+
    wordcode but is retained as the ``opc.get_jump_target_maps`` entry
    point that opcode modules have historically exposed. Fallthrough
    edges are only recorded between physically adjacent instructions, so
    inline ``CACHE`` words (3.11+) and jump targets computed in the
    wrong unit do not invent or drop edges. An ``EXTENDED_ARG`` chain is
    one logical instruction located at its first word.
    """
    unpack_opargs = (
        unpack_opargs_wordcode
        if opc.version_tuple < (3, 10)
        else unpack_opargs_bytecode_310
    )
    extended_arg_op = getattr(opc, "EXTENDED_ARG", None)

    offset2prev = {}
    prev_node = -1
    prev_end = -1
    chain_head = -1
    for offset, op, arg in unpack_opargs(code, opc):
        if op == extended_arg_op:
            if chain_head < 0:
                chain_head = offset
            continue
        node = chain_head if chain_head >= 0 else offset
        chain_head = -1
        if prev_node >= 0 and prev_end == node:
            prev_list = offset2prev.get(node, [])
            prev_list.append(prev_node)
            offset2prev[node] = prev_list
        if falls_through(op, opc):
            prev_node = node
        else:
            prev_node = -1
        prev_end = (
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
