"""Unit tests for xdis.bytecode_entry: multi-entry (mid-instruction and
mid-EXTENDED_ARG-chain) bytecode decoding used against jump-obfuscation
anti-analysis."""

import unittest
from types import SimpleNamespace

from xdis.bytecode_entry import (
    EntryDecodeError,
    EntryKind,
    classify_entry,
    decode_entries,
    decode_entry,
    encoding_rules,
    find_entry_offsets,
    linear_scan,
    version_encoding_table,
)
from xdis.opcodes import opcode_27, opcode_36


def op(opc, name):
    return opc.opmap[name]


class TestEncodingRules(unittest.TestCase):
    """Compare instruction width, EXTENDED_ARG encoding, and entry
    recovery across CPython versions."""

    def test_pre_36_variable_length(self):
        rules = encoding_rules(opcode_27)
        self.assertFalse(rules.fixed_wordsize)
        self.assertEqual((rules.no_arg_size, rules.arg_size), (1, 3))
        self.assertEqual(rules.extended_arg_shift, 16)
        self.assertEqual(rules.jump_offset_unit, 1)
        # variable-length streams can land back on a canonical boundary
        self.assertTrue(rules.resync_possible)
        self.assertFalse(rules.has_cache_slots)

    def test_36_wordcode(self):
        rules = encoding_rules(opcode_36)
        self.assertTrue(rules.fixed_wordsize)
        self.assertEqual((rules.no_arg_size, rules.arg_size), (2, 2))
        self.assertEqual(rules.extended_arg_shift, 8)
        self.assertEqual(rules.jump_offset_unit, 1)
        # fixed 2-byte parity: an odd entry can never resynchronize
        self.assertFalse(rules.resync_possible)

    def test_version_table_spans_encoding_changes(self):
        table = {r.version_tuple[:2]: r for r in version_encoding_table()}
        self.assertEqual(table[(2, 7)].extended_arg_shift, 16)
        self.assertEqual(table[(3, 6)].extended_arg_shift, 8)
        self.assertEqual(table[(3, 9)].jump_offset_unit, 1)
        self.assertEqual(table[(3, 10)].jump_offset_unit, 2)
        self.assertFalse(table[(3, 10)].has_cache_slots)
        self.assertTrue(table[(3, 11)].has_cache_slots)
        self.assertTrue(table[(3, 5)].resync_possible)
        self.assertFalse(table[(3, 6)].resync_possible)


class TestLinearScan36(unittest.TestCase):
    """3.6+ wordcode: fixed 2-byte instructions."""

    def setUp(self):
        self.opc = opcode_36
        EXTENDED_ARG = op(self.opc, "EXTENDED_ARG")
        LOAD_CONST = op(self.opc, "LOAD_CONST")
        RETURN_VALUE = op(self.opc, "RETURN_VALUE")
        # EXTENDED_ARG 1 ; LOAD_CONST 0 ;; RETURN_VALUE
        self.bytecode = bytes(
            [EXTENDED_ARG, 1, LOAD_CONST, 0, RETURN_VALUE, 0]
        )

    def test_linear_scan_boundaries(self):
        scan = linear_scan(self.bytecode, self.opc)
        self.assertEqual(scan.physical_starts, (0, 2, 4))
        self.assertEqual(scan.chain_continuations, (2,))
        self.assertEqual(scan.logical_starts, {0, 4})
        self.assertIsNone(scan.truncated_at)

    def test_classify(self):
        opc = self.opc
        bc = self.bytecode
        self.assertEqual(classify_entry(bc, 0, opc), EntryKind.LEGAL)
        self.assertEqual(classify_entry(bc, 2, opc), EntryKind.CHAIN_CONTINUATION)
        self.assertEqual(classify_entry(bc, 4, opc), EntryKind.LEGAL)
        # odd offsets land on operand bytes of the fixed-width words
        self.assertEqual(classify_entry(bc, 1, opc), EntryKind.OVERLAPPING)
        self.assertEqual(classify_entry(bc, 3, opc), EntryKind.OVERLAPPING)
        self.assertEqual(classify_entry(bc, 6, opc), EntryKind.OUT_OF_BOUNDS)
        self.assertEqual(classify_entry(bc, 100, opc), EntryKind.OUT_OF_BOUNDS)
        self.assertEqual(classify_entry(bc, -2, opc), EntryKind.OUT_OF_BOUNDS)

    def test_legal_entry_matches_linear_sweep(self):
        got = [
            (ei.instruction.offset, ei.instruction.opname, ei.instruction.arg)
            for ei in decode_entry(self.bytecode, 0, self.opc)
        ]
        # Linear sweep combines EXTENDED_ARG into the LOAD_CONST operand
        self.assertEqual(
            got,
            [
                (0, "EXTENDED_ARG", 1),
                (2, "LOAD_CONST", 0x100),
                (4, "RETURN_VALUE", None),
            ],
        )
        stream = decode_entry(self.bytecode, 0, self.opc)
        self.assertTrue(all(ei.resynchronized for ei in stream))

    def test_chain_continuation_entry_restarts_extended_arg(self):
        """Jumping into the middle of an EXTENDED_ARG chain restarts the
        accumulation: the executed operand is the unextended one."""
        result = decode_entry(self.bytecode, 2, self.opc)
        self.assertEqual(result[0].entry_kind, EntryKind.CHAIN_CONTINUATION)
        inst = result[0].instruction
        self.assertEqual(inst.opname, "LOAD_CONST")
        # Fresh EXTENDED_ARG state: arg is 0, not 0x100 as in the sweep
        self.assertEqual(inst.arg, 0)
        self.assertTrue(result[0].mid_instruction)
        self.assertFalse(result[0].resynchronized)
        # The stream rejoins canonical boundaries at RETURN_VALUE (offset 4)
        self.assertEqual(result[-1].instruction.offset, 4)
        self.assertTrue(result[-1].resynchronized)

    def test_overlapping_entry_is_flagged_not_silent(self):
        """An operand byte that happens to be a harmless opcode (NOP) must
        be flagged as mid-instruction data, not silently reported as a
        safe instruction."""
        NOP = op(self.opc, "NOP")
        LOAD_CONST = op(self.opc, "LOAD_CONST")
        RETURN_VALUE = op(self.opc, "RETURN_VALUE")
        # LOAD_CONST <NOP-as-operand> ; RETURN_VALUE
        bytecode = bytes([LOAD_CONST, NOP, RETURN_VALUE, 0])
        result = decode_entry(bytecode, 1, self.opc)
        self.assertEqual(result[0].entry_kind, EntryKind.OVERLAPPING)
        self.assertEqual(result[0].instruction.opname, "NOP")
        self.assertTrue(result[0].mid_instruction)
        # Fixed wordcode parity: an odd entry never resynchronizes
        self.assertFalse(any(ei.resynchronized for ei in result))

    def test_out_of_bounds_entry_raises(self):
        with self.assertRaises(EntryDecodeError):
            decode_entry(self.bytecode, len(self.bytecode), self.opc)

    def test_truncated_bytecode_raises(self):
        # A lone opcode byte without its operand byte
        LOAD_CONST = op(self.opc, "LOAD_CONST")
        with self.assertRaises(EntryDecodeError):
            decode_entry(bytes([LOAD_CONST]), 0, self.opc)


class TestMultiEntry36(unittest.TestCase):
    """decode_entries over all jump targets of obfuscated wordcode."""

    def test_jump_into_extended_arg_chain(self):
        opc = opcode_36
        EXTENDED_ARG = op(opc, "EXTENDED_ARG")
        LOAD_CONST = op(opc, "LOAD_CONST")
        JUMP_ABSOLUTE = op(opc, "JUMP_ABSOLUTE")
        RETURN_VALUE = op(opc, "RETURN_VALUE")
        # 0: EXTENDED_ARG 1
        # 2: LOAD_CONST 0      <- jump target, mid-chain
        # 4: JUMP_ABSOLUTE 2
        # 6: RETURN_VALUE
        bytecode = bytes(
            [EXTENDED_ARG, 1, LOAD_CONST, 0, JUMP_ABSOLUTE, 2, RETURN_VALUE, 0]
        )
        results = {r.entry_offset: r for r in decode_entries(bytecode, opc)}
        self.assertIn(0, results)
        self.assertIn(2, results)
        entry2 = results[2]
        self.assertEqual(entry2.kind, EntryKind.CHAIN_CONTINUATION)
        self.assertIsNone(entry2.error)
        # first instruction is flagged; stream resyncs at offset 4
        self.assertTrue(entry2.instructions[0].mid_instruction)
        self.assertEqual(entry2.resync_offset, 4)
        # legal entry 0 needs no recovery
        self.assertEqual(results[0].kind, EntryKind.LEGAL)
        self.assertIsNone(results[0].resync_offset)

    def test_out_of_bounds_entry_recorded_and_strict(self):
        opc = opcode_36
        RETURN_VALUE = op(opc, "RETURN_VALUE")
        bytecode = bytes([RETURN_VALUE, 0])
        results = decode_entries(bytecode, opc, entries=[0, 50])
        self.assertEqual(results[1].kind, EntryKind.OUT_OF_BOUNDS)
        self.assertIsNotNone(results[1].error)
        self.assertEqual(results[1].instructions, ())
        with self.assertRaises(EntryDecodeError):
            decode_entries(bytecode, opc, entries=[50], strict=True)

    def test_find_entry_offsets(self):
        opc = opcode_36
        JUMP_ABSOLUTE = op(opc, "JUMP_ABSOLUTE")
        RETURN_VALUE = op(opc, "RETURN_VALUE")
        bytecode = bytes([JUMP_ABSOLUTE, 4, RETURN_VALUE, 0, RETURN_VALUE, 0])
        co = SimpleNamespace(co_code=bytecode)
        self.assertEqual(find_entry_offsets(co, opc), [0, 4])


class TestVariableLength27(unittest.TestCase):
    """Pre-3.6: 1-or-3-byte instructions, 16-bit EXTENDED_ARG shift, and
    resynchronization of misaligned streams."""

    def setUp(self):
        self.opc = opcode_27
        EXTENDED_ARG = op(self.opc, "EXTENDED_ARG")
        LOAD_CONST = op(self.opc, "LOAD_CONST")
        RETURN_VALUE = op(self.opc, "RETURN_VALUE")
        # 0: EXTENDED_ARG 1 (3 bytes)
        # 3: LOAD_CONST 0  (3 bytes) <- chain continuation
        # 6: RETURN_VALUE  (1 byte)
        self.bytecode = bytes(
            [EXTENDED_ARG, 1, 0, LOAD_CONST, 0, 0, RETURN_VALUE]
        )

    def test_classify(self):
        opc = self.opc
        bc = self.bytecode
        self.assertEqual(classify_entry(bc, 0, opc), EntryKind.LEGAL)
        # operand bytes of the 3-byte EXTENDED_ARG
        self.assertEqual(classify_entry(bc, 1, opc), EntryKind.OVERLAPPING)
        self.assertEqual(classify_entry(bc, 2, opc), EntryKind.OVERLAPPING)
        self.assertEqual(classify_entry(bc, 3, opc), EntryKind.CHAIN_CONTINUATION)
        self.assertEqual(classify_entry(bc, 6, opc), EntryKind.LEGAL)
        self.assertEqual(classify_entry(bc, 7, opc), EntryKind.OUT_OF_BOUNDS)

    def test_chain_continuation_uses_fresh_extended_arg(self):
        result = decode_entry(self.bytecode, 3, self.opc)
        inst = result[0].instruction
        self.assertEqual(inst.opname, "LOAD_CONST")
        # Linear sweep sees operand 0x10000; a jump here executes 0
        self.assertEqual(inst.arg, 0)
        self.assertTrue(result[0].mid_instruction)
        self.assertEqual(result[-1].instruction.opname, "RETURN_VALUE")
        self.assertTrue(result[-1].resynchronized)

    def test_misaligned_stream_resynchronizes(self):
        """With the variable 1/3-byte encoding, a stream entered on operand
        bytes can land back on a canonical instruction boundary."""
        opc = self.opc
        NOP = op(opc, "NOP")
        LOAD_CONST = op(opc, "LOAD_CONST")
        RETURN_VALUE = op(opc, "RETURN_VALUE")
        # 0: LOAD_CONST with operand bytes [NOP, RETURN_VALUE]
        # 3: RETURN_VALUE
        bytecode = bytes([LOAD_CONST, NOP, RETURN_VALUE, RETURN_VALUE])
        result = decode_entry(bytecode, 1, opc)
        offsets = [(ei.instruction.offset, ei.instruction.opname,
                    ei.mid_instruction, ei.resynchronized) for ei in result]
        self.assertEqual(
            offsets,
            [
                (1, "NOP", True, False),
                (2, "RETURN_VALUE", True, False),
                (3, "RETURN_VALUE", False, True),
            ],
        )

    def test_decode_entries_records_resync(self):
        opc = self.opc
        NOP = op(opc, "NOP")
        LOAD_CONST = op(opc, "LOAD_CONST")
        RETURN_VALUE = op(opc, "RETURN_VALUE")
        bytecode = bytes([LOAD_CONST, NOP, RETURN_VALUE, RETURN_VALUE])
        results = {r.entry_offset: r for r in decode_entries(bytecode, opc, entries=[1])}
        self.assertEqual(results[1].kind, EntryKind.OVERLAPPING)
        self.assertEqual(results[1].resync_offset, 3)


if __name__ == "__main__":
    unittest.main()
