# cython: language_level=3

from bindings cimport human as c

cdef class Human:
    cdef c.human_t* _h

    def __cinit__(self):
        self._h = NULL

    def __dealloc__(self):
        if self._h is not NULL:
            c.human_free(self._h)

    @staticmethod
    def load(path):
        cdef Human inst = Human()
        cdef bytes b = path.encode("utf-8") if isinstance(path, str) else path
        cdef c.human_status_t st = c.human_load(b, &inst._h)
        if st != c.HUMAN_OK:
            raise IOError(f"human_load failed: {int(st)}")
        return inst

    @property
    def vertex_count(self):
        return int(c.human_vertex_count(self._h))

    @property
    def index_count(self):
        return int(c.human_index_count(self._h))

    @property
    def param_count(self):
        return int(c.human_param_count(self._h))

    def set_param(self, uint_id, float value):
        cdef c.human_status_t st = c.human_set_param(self._h, uint_id, value)
        if st != c.HUMAN_OK:
            raise ValueError(f"set_param({uint_id}) failed: {int(st)}")

    def get_param(self, uint_id):
        return float(c.human_get_param(self._h, uint_id))

    def evaluate(self):
        cdef c.human_status_t st = c.human_evaluate(self._h)
        if st != c.HUMAN_OK:
            raise RuntimeError(f"evaluate failed: {int(st)}")
