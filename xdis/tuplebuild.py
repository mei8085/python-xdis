# Copyright (c) 2024-2026 by Rocky Bernstein
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

"""
Identity-preserving construction of immutable tuples while unmarshalling.

The marshal format (version 3 and later, i.e. Python 3.4+) serializes an
object reference graph: a container that is written with ``FLAG_REF`` is
assigned a reference slot *before* its contents are written, and a later
``TYPE_REF`` tag points back at that slot.  Replicating this on load means
the exact Python object placed in the reference table has to remain the
object that eventually gets returned -- even for immutable tuples, whose
children (including the tuple itself through a back reference) are read
before the tuple is fully built.

CPython's ``marshal.c`` solves this by allocating an empty tuple of the
final size and filling its slots in place via internal C APIs.  Ordinary
Python code cannot mutate a tuple, so on CPython we use ``ctypes`` to do
the same thing safely (writing only into the initially NULL slots, with
matching ``Py_IncRef`` calls).  On interpreters where ``ctypes`` access to
the C API is unavailable (PyPy, GraalPy, ...), we fall back to building a
plain tuple from a temporary list; every non-recursive stream is
reconstructed identically there, while tuple-only cycles lose object
identity (the back reference resolves to the temporary list), which is
the closest pure-Python approximation possible without mutable tuples.
"""

import sys

#: True when the running interpreter lays tuples out the way CPython does
#: and exposes the C entry points we need to fill tuple slots in place.
_NATIVE_TUPLE_FILL = False

# CPython only: PyPy/GraalPy do not provide ``ctypes.pythonapi`` with the
# CPython object layout, and their tuples are not a flat C array.
if sys.implementation.name == "cpython":
    try:
        import ctypes

        _pythonapi = ctypes.pythonapi

        _PyTuple_New = _pythonapi.PyTuple_New
        _PyTuple_New.argtypes = [ctypes.c_ssize_t]
        _PyTuple_New.restype = ctypes.py_object

        _Py_IncRef = _pythonapi.Py_IncRef
        _Py_IncRef.argtypes = [ctypes.py_object]
        _Py_IncRef.restype = None

        # PyTupleObject starts with three machine-sized header words
        # (ob_refcnt, ob_type, ob_size); the item array follows inline.
        # The pointer width is used rather than a hard-coded offset so
        # that 32-bit builds keep working.
        _SLOT_OFFSET = 3 * ctypes.sizeof(ctypes.c_void_p)
        _POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)

        def _write_tuple_slot(tuple_obj: tuple, index: int, value) -> None:
            # The destination slot is still the NULL pointer left by
            # PyTuple_New, so writing into it does not overwrite (and
            # therefore leak) a previous reference.  We do owe the stored
            # pointer one reference, hence the explicit incref.
            _Py_IncRef(value)
            ctypes.c_void_p.from_address(
                id(tuple_obj) + _SLOT_OFFSET + index * _POINTER_SIZE
            ).value = id(value)

        # Smoke-test the mechanism once, at import time: if anything about
        # the layout differs from our assumptions, disable the fast path
        # instead of risking memory corruption later.
        _probe = _PyTuple_New(1)
        _write_tuple_slot(_probe, 0, None)
        if _probe != (None,):  # pragma: no cover - only on an ABI mismatch
            raise RuntimeError("tuple slot layout probe failed")
        del _probe
        _NATIVE_TUPLE_FILL = True
    except Exception:  # pragma: no cover - non-CPython or restricted builds
        _NATIVE_TUPLE_FILL = False


class TupleBuilder:
    """
    Accumulates the elements of a marshal tuple while keeping a stable
    placeholder object in the unmarshaller's reference table.

    On CPython the placeholder *is* the final tuple: slots are filled in
    place as child objects are read, so a ``TYPE_REF`` pointing at the
    tuple while it is still being constructed resolves to the very object
    that is eventually returned.  This is what makes self-referential and
    mutually recursive tuples deserialize into the same graph CPython
    produces.

    On other interpreters the placeholder is a temporary list that
    :meth:`finish` converts into the returned tuple; back references fired
    while construction was still in progress resolve to that list.
    """

    __slots__ = ("placeholder", "size", "_slots_written")

    def __init__(self, size: int):
        self.size = size
        self._slots_written = 0
        if _NATIVE_TUPLE_FILL:
            # An all-NULL tuple of the final size; its identity never changes.
            self.placeholder = _PyTuple_New(size)
        else:
            # Mutable stand-in, replaced by a real tuple in finish().
            self.placeholder = []

    def append(self, value) -> None:
        if _NATIVE_TUPLE_FILL:
            _write_tuple_slot(self.placeholder, self._slots_written, value)
        else:
            self.placeholder.append(value)
        self._slots_written += 1

    def finish(self) -> tuple:
        if _NATIVE_TUPLE_FILL:
            # Every slot was written in place; the placeholder already is
            # the finished tuple and keeps its identity and reference slot.
            return self.placeholder
        return tuple(self.placeholder)
