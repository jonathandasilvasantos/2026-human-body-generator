# cython: language_level=3
# distutils: language = c

from libc.stdint cimport uint32_t

cimport bindings.human as c

import numpy as np
cimport numpy as cnp

cnp.import_array()


_STATUS_NAMES = {
    c.HUMAN_OK: "OK",
    c.HUMAN_ERR_IO: "IO",
    c.HUMAN_ERR_FORMAT: "FORMAT",
    c.HUMAN_ERR_OOM: "OOM",
    c.HUMAN_ERR_RANGE: "RANGE",
}


def _check(c.human_status_t st, op):
    if st != c.HUMAN_OK:
        raise RuntimeError(f"{op} -> {_STATUS_NAMES.get(int(st), int(st))}")


cdef class Human:
    cdef c.human_t* _h

    def __cinit__(self):
        self._h = NULL

    def __dealloc__(self):
        if self._h is not NULL:
            c.human_free(self._h)
            self._h = NULL

    @staticmethod
    def load(path):
        cdef Human inst = Human()
        cdef bytes b = str(path).encode("utf-8") if not isinstance(path, (bytes, bytearray)) else bytes(path)
        _check(c.human_load(b, &inst._h), "load")
        return inst

    @staticmethod
    def proto(archetype=c.HUMAN_ARCH_PROTO):
        cdef Human inst = Human()
        _check(c.human_create_proto(<uint32_t>archetype, &inst._h), "create_proto")
        return inst

    def save(self, path):
        cdef bytes b = str(path).encode("utf-8") if not isinstance(path, (bytes, bytearray)) else bytes(path)
        _check(c.human_save(self._h, b), "save")

    def evaluate(self):
        _check(c.human_evaluate(self._h), "evaluate")

    @property
    def vertex_count(self):
        return int(c.human_vertex_count(self._h))

    @property
    def index_count(self):
        return int(c.human_index_count(self._h))

    @property
    def bone_count(self):
        return int(c.human_bone_count(self._h))

    @property
    def vertices(self):
        """(N, 8) float32 view: pos.xyz, norm.xyz, uv.xy. Zero-copy."""
        cdef uint32_t n = c.human_vertex_count(self._h)
        cdef const float* p = c.human_vertex_buffer(self._h)
        cdef cnp.npy_intp shape[2]
        shape[0] = <cnp.npy_intp>n
        shape[1] = 8
        cdef cnp.ndarray arr = cnp.PyArray_SimpleNewFromData(
            2, shape, cnp.NPY_FLOAT32, <void*>p)
        cnp.PyArray_SetBaseObject(arr, self)
        cnp.Py_INCREF(self)
        return arr

    @property
    def indices(self):
        cdef uint32_t n = c.human_index_count(self._h)
        cdef const uint32_t* p = c.human_index_buffer(self._h)
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp>n
        cdef cnp.ndarray arr = cnp.PyArray_SimpleNewFromData(
            1, shape, cnp.NPY_UINT32, <void*>p)
        cnp.PyArray_SetBaseObject(arr, self)
        cnp.Py_INCREF(self)
        return arr

    @property
    def bone_matrices(self):
        cdef uint32_t n = c.human_bone_count(self._h)
        cdef const float* p = c.human_bone_world_matrices(self._h)
        cdef cnp.npy_intp shape[3]
        shape[0] = <cnp.npy_intp>n
        shape[1] = 4
        shape[2] = 4
        cdef cnp.ndarray arr = cnp.PyArray_SimpleNewFromData(
            3, shape, cnp.NPY_FLOAT32, <void*>p)
        cnp.PyArray_SetBaseObject(arr, self)
        cnp.Py_INCREF(self)
        return arr

    @property
    def params(self):
        """List of (name, kind, min, max, default, value) tuples."""
        cdef uint32_t n = c.human_param_count(self._h)
        cdef uint32_t i
        cdef float lo, hi, df
        cdef const char* nm
        out = []
        for i in range(n):
            nm = c.human_param_name(self._h, i)
            _check(c.human_param_range(self._h, i, &lo, &hi, &df), "param_range")
            kind = "morph" if int(c.human_param_kind(self._h, i)) == int(c.HUMAN_PARAM_MORPH) else "bone_scale_y"
            out.append((
                nm.decode("utf-8") if nm != NULL else f"p{i}",
                kind,
                float(lo), float(hi), float(df),
                float(c.human_get_param(self._h, i)),
            ))
        return out

    def set_param(self, ident, float value):
        cdef uint32_t pid = self._resolve(ident)
        _check(c.human_set_param(self._h, pid, value), "set_param")

    def get_param(self, ident):
        cdef uint32_t pid = self._resolve(ident)
        return float(c.human_get_param(self._h, pid))

    cdef uint32_t _resolve(self, ident) except? 0xFFFFFFFF:
        cdef uint32_t i, n
        cdef const char* nm
        if isinstance(ident, int):
            return <uint32_t>ident
        target = ident.encode("utf-8") if isinstance(ident, str) else ident
        n = c.human_param_count(self._h)
        for i in range(n):
            nm = c.human_param_name(self._h, i)
            if nm != NULL and nm == <const char*>target:
                return i
            if nm != NULL and nm[0] != 0 and target == bytes(nm):
                return i
        raise KeyError(f"unknown param: {ident!r}")
