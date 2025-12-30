#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(MLDSA44)
    #include "mldsa44/polyvec.h"
    #include "mldsa44/params.h"
    #include "mldsa44/poly.h"
    #include "mldsa44/reduce.h"
    #include "mldsa44/fips202.h"
    #include "mldsa44/rounding.h"
    #include "mldsa44/symmetric.h"
    #include "mldsa44/ntt.h"
#elif defined(MLDSA65)
    #include "mldsa65/polyvec.h"
    #include "mldsa65/params.h"
    #include "mldsa65/poly.h"
    #include "mldsa65/reduce.h"
    #include "mldsa65/fips202.h"
    #include "mldsa65/rounding.h"
    #include "mldsa65/symmetric.h"
    #include "mldsa65/ntt.h"
#elif defined(MLDSA87)
    #include "mldsa87/polyvec.h"
    #include "mldsa87/params.h"
    #include "mldsa87/poly.h"
    #include "mldsa87/reduce.h"
    #include "mldsa87/fips202.h"
    #include "mldsa87/rounding.h"
    #include "mldsa87/symmetric.h"
    #include "mldsa87/ntt.h"
#else
    #error "Define one of MLDSA44, MLDSA65, or MLDSA87"
#endif


static PyObject* py_sample_s1(PyObject* self, PyObject* args) {
    PyObject *seed_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &seed_list)) {
        PyErr_SetString(PyExc_TypeError, "argument must be a list");
        return NULL;
    }

    Py_ssize_t len = PyList_Size(seed_list);
    if (len != CRHBYTES) {
        PyErr_SetString(PyExc_ValueError, "seed length must be CRHBYTES");
        return NULL;
    }

    uint8_t seed[CRHBYTES];
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *item = PyList_GetItem(seed_list, i);
        long value = PyLong_AsLong(item);
        if (value < 0 || value > 255) {
            PyErr_SetString(PyExc_ValueError, "seed elements must be in [0, 255]");
            return NULL;
        }
        seed[i] = (uint8_t)value;
    }

    polyvecl s1;
    #if defined(MLDSA44)
        PQCLEAN_MLDSA44_CLEAN_polyvecl_uniform_eta(&s1, seed, 0);
    #elif defined(MLDSA65)
        PQCLEAN_MLDSA65_CLEAN_polyvecl_uniform_eta(&s1, seed, 0);
    #elif defined(MLDSA87)
        PQCLEAN_MLDSA87_CLEAN_polyvecl_uniform_eta(&s1, seed, 0);
    #endif

    PyObject* pylist = PyList_New(L);
    for (int i = 0; i < L; i++) {
        PyObject* coeffs = PyList_New(N);
        for (int j = 0; j < N; j++) {
            PyList_SetItem(coeffs, j, PyLong_FromLong(s1.vec[i].coeffs[j]));
        }
        PyList_SetItem(pylist, i, coeffs);
    }
    return pylist;
}

static PyObject* py_sample_c(PyObject* self, PyObject* args) {
    PyObject *seed_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &seed_list)) {
        PyErr_SetString(PyExc_TypeError, "argument must be a list");
        return NULL;
    }

    Py_ssize_t len = PyList_Size(seed_list);
    if (len != CTILDEBYTES) {
        PyErr_SetString(PyExc_ValueError, "seed length must be CTILDEBYTES");
        return NULL;
    }

    uint8_t seed[CTILDEBYTES];
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *item = PyList_GetItem(seed_list, i);
        long value = PyLong_AsLong(item);
        if (value < 0 || value > 255) {
            PyErr_SetString(PyExc_ValueError, "seed elements must be in [0, 255]");
            return NULL;
        }
        seed[i] = (uint8_t)value;
    }

    poly cp;
    #if defined(MLDSA44)
        PQCLEAN_MLDSA44_CLEAN_poly_challenge(&cp, seed);
    #elif defined(MLDSA65)
        PQCLEAN_MLDSA65_CLEAN_poly_challenge(&cp, seed);
    #elif defined(MLDSA87)
        PQCLEAN_MLDSA87_CLEAN_poly_challenge(&cp, seed);
    #endif

    PyObject* coeffs = PyList_New(N);
    for (int j = 0; j < N; j++) {
        PyList_SetItem(coeffs, j, PyLong_FromLong(cp.coeffs[j]));
    }
    return coeffs;
}

static PyObject* py_NTT(PyObject* self, PyObject* args) {
    PyObject *seed_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &seed_list)) {
        PyErr_SetString(PyExc_TypeError, "argument must be a list");
        return NULL;
    }

    Py_ssize_t len = PyList_Size(seed_list);
    if (len != N) {
        PyErr_SetString(PyExc_ValueError, "list length must be N");
        return NULL;
    }

    int32_t a[N];
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *item = PyList_GetItem(seed_list, i);
        long value = PyLong_AsLong(item);
        a[i] = (int32_t)value;
    }

    #if defined(MLDSA44)
        PQCLEAN_MLDSA44_CLEAN_ntt(a);
    #elif defined(MLDSA65)
        PQCLEAN_MLDSA65_CLEAN_ntt(a);
    #elif defined(MLDSA87)
        PQCLEAN_MLDSA87_CLEAN_ntt(a);
    #endif

    PyObject* coeffs = PyList_New(N);
    for (int j = 0; j < N; j++) {
        PyList_SetItem(coeffs, j, PyLong_FromLong(a[j]));
    }
    return coeffs;
}

static PyObject* py_montgomery_reduce(PyObject* self, PyObject* args) {
    long long a;  
    if (!PyArg_ParseTuple(args, "L", &a)) { 
        return NULL;
    }

    int32_t result;
    #if defined(MLDSA44)
        result = PQCLEAN_MLDSA44_CLEAN_montgomery_reduce((int64_t)a);
    #elif defined(MLDSA65)
        result = PQCLEAN_MLDSA65_CLEAN_montgomery_reduce((int64_t)a);
    #elif defined(MLDSA87)
        result = PQCLEAN_MLDSA87_CLEAN_montgomery_reduce((int64_t)a);
    #endif

    return PyLong_FromLong((long)result);
}

static PyMethodDef Methods[] = {
    {"sample_s1", py_sample_s1, METH_VARARGS, "Generate s1 vector from seed"},
    {"sample_c", py_sample_c, METH_VARARGS, "Generate c poly from seed"},
    {"NTT", py_NTT, METH_VARARGS, "NTT"},
    {"montgomery_reduce", py_montgomery_reduce, METH_VARARGS, "montgomery_reduce"},
    {NULL, NULL, 0, NULL}
};

#if defined(MLDSA44)
static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "mldsa44", NULL, -1, Methods};
PyMODINIT_FUNC PyInit_mldsa44(void) { return PyModule_Create(&moduledef); }

#elif defined(MLDSA65)
static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "mldsa65", NULL, -1, Methods};
PyMODINIT_FUNC PyInit_mldsa65(void) { return PyModule_Create(&moduledef); }

#elif defined(MLDSA87)
static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "mldsa87", NULL, -1, Methods};
PyMODINIT_FUNC PyInit_mldsa87(void) { return PyModule_Create(&moduledef); }
#endif