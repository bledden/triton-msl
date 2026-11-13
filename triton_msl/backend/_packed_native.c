#define PY_SSIZE_T_CLEAN
#include <Python.h>
#if PY_VERSION_HEX >= 0x030E0000 && PY_VERSION_HEX < 0x030F0000 && !defined(Py_GIL_DISABLED)
#define Py_BUILD_CORE
#include "internal/pycore_dict.h"
#undef Py_BUILD_CORE
#define PACKED_UNICODE_KEYS 1
#endif

enum { CHILD, FALLBACK, REFUSE, PYTHON_LEAF, KEY_COUNT };
static const char *provider_names[KEY_COUNT] = {
    "_checked_packed_copy", "_UsePackedJSON", "_refuse", "_checked_packed_copy_python"
};
typedef struct { PyObject *keys[KEY_COUNT]; } State;
typedef struct { PyObject *globals; State *state; } Context;

/* Preserve exceptions raised by dictionary-key comparisons. The String API
   deliberately suppresses those exceptions and cannot implement LOAD_GLOBAL. */
static PyObject *lookup(Context *c, int index) {
    PyObject *key = NULL;
#ifdef PACKED_UNICODE_KEYS
    /* The current representation excludes custom/subclass keys. No callback,
       allocation or GIL release occurs before the live lookup. Recheck after
       every recursive edge: a callback may replace the entire key domain. */
    if (DK_IS_UNICODE(((PyDictObject *)c->globals)->ma_keys))
        key = Py_NewRef(c->state->keys[index]);
#endif
    /* General dictionaries retain fresh identity for colliding foreign keys. */
    if (!key) key = PyUnicode_FromString(provider_names[index]);
    if (!key) return NULL;
    PyObject *out = PyDict_GetItemWithError(c->globals, key);
    Py_XINCREF(out);
    Py_DECREF(key);
    return out;
}

/* Each Python recursive edge resolves the live helper after earlier callbacks.
   Keep that call boundary, including replacement and in-place code changes. */
static PyObject *child_copy(Context *c, PyObject *value, PyObject *plan) {
    PyObject *fn = lookup(c, CHILD);
    if (!fn) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_NameError, "name '_checked_packed_copy' is not defined");
        return NULL;
    }
    PyObject *args[2] = {value, plan};
    PyObject *out = PyObject_Vectorcall(fn, args, 2, NULL);
    Py_DECREF(fn);
    return out;
}

static PyObject *fallback(Context *c) {
    PyObject *kind = lookup(c, FALLBACK);
    if (kind) { PyErr_SetNone(kind); Py_DECREF(kind); }
    else if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "missing JSON fallback exception");
    return NULL;
}
static PyObject *refuse(Context *c) {
    PyObject *fn = lookup(c, REFUSE);
    if (!fn) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "missing refusal provider"); return NULL; }
    PyObject *out = PyObject_CallFunction(fn, "s", "descriptor changed after launcher construction");
    Py_DECREF(fn);
    return out;
}
static PyObject *visit(Context *c, PyObject *value, PyObject *plan);
static PyObject *visit_body(Context *c, PyObject *value, PyObject *plan) {
    if (!value) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "missing descriptor value");
        return NULL;
    }
    if (!PyTuple_CheckExact(plan) || PyTuple_GET_SIZE(plan) != 2) {
        PyErr_SetString(PyExc_ValueError, "malformed private descriptor plan"); return NULL;
    }
    PyObject *kind = PyTuple_GET_ITEM(plan, 0), *wanted = PyTuple_GET_ITEM(plan, 1);
    if ((kind==(PyObject *)&PyList_Type || kind==(PyObject *)&PyDict_Type) && !PyTuple_CheckExact(wanted)) {
        PyErr_SetString(PyExc_ValueError,"noncanonical private container plan"); return NULL;
    }
    if (kind == (PyObject *)&PyList_Type) {
        if (!PyList_CheckExact(value) && !PyTuple_CheckExact(value)) return fallback(c);
        PyObject *observed = PyList_CheckExact(value) ? PyList_AsTuple(value) : Py_NewRef(value);
        if (!observed) return NULL;
        Py_ssize_t n = PyTuple_Size(wanted);
        if (n < 0) { Py_DECREF(observed); return NULL; }
        if (PyTuple_GET_SIZE(observed) != n) { Py_DECREF(observed); return refuse(c); }
        PyObject *result = PyList_New(n);
        if (!result) { Py_DECREF(observed); return NULL; }
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *child = child_copy(c, PyTuple_GET_ITEM(observed,i), PyTuple_GET_ITEM(wanted,i));
            if (!child) { Py_DECREF(result); Py_DECREF(observed); return NULL; }
            PyList_SET_ITEM(result,i,child);
        }
        Py_DECREF(observed);
        return result;
    }
    if (kind == (PyObject *)&PyDict_Type) {
        if (!PyDict_CheckExact(value)) return fallback(c);
        PyObject *observed = PyDict_Copy(value);
        if (!observed) return NULL;
        Py_ssize_t pos = 0; PyObject *key, *item;
        while (PyDict_Next(observed,&pos,&key,&item)) {
            if (!PyUnicode_CheckExact(key)) { Py_DECREF(observed); return fallback(c); }
        }
        Py_ssize_t n = PyTuple_Size(wanted);
        if (n < 0) { Py_DECREF(observed); return NULL; }
        if (PyDict_Size(observed) != n) { Py_DECREF(observed); return fallback(c); }
        /* Check the complete key set before inspecting any child, as Python does. */
        for (Py_ssize_t i=0;i<n;i++) {
            PyObject *pair=PyTuple_GET_ITEM(wanted,i);
            if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair)!=2) {
                Py_DECREF(observed); PyErr_SetString(PyExc_ValueError,"malformed key plan"); return NULL;
            }
            if (!PyUnicode_CheckExact(PyTuple_GET_ITEM(pair,0))) {
                Py_DECREF(observed); PyErr_SetString(PyExc_ValueError,"non-string private key plan"); return NULL;
            }
            int found=PyDict_Contains(observed,PyTuple_GET_ITEM(pair,0));
            if (found<0) { Py_DECREF(observed); return NULL; }
            if (!found) { Py_DECREF(observed); return fallback(c); }
        }
        PyObject *result=PyDict_New();
        if (!result) { Py_DECREF(observed); return NULL; }
        for (Py_ssize_t i=0;i<n;i++) {
            PyObject *pair=PyTuple_GET_ITEM(wanted,i), *key=PyTuple_GET_ITEM(pair,0);
            PyObject *child=child_copy(c,PyDict_GetItemWithError(observed,key),PyTuple_GET_ITEM(pair,1));
            if (!child || PyDict_SetItem(result,key,child)<0) {
                Py_XDECREF(child); Py_DECREF(result); Py_DECREF(observed); return NULL;
            }
            Py_DECREF(child);
        }
        Py_DECREF(observed); return result;
    }
    if ((PyObject *)Py_TYPE(wanted) != kind ||
            (kind!=(PyObject *)&PyUnicode_Type && kind!=(PyObject *)&PyLong_Type &&
             kind!=(PyObject *)&PyFloat_Type && kind!=(PyObject *)&PyBool_Type &&
             wanted!=Py_None)) {
        PyErr_SetString(PyExc_ValueError,"noncanonical private leaf plan"); return NULL;
    }
    if ((PyObject *)Py_TYPE(value) != kind) return fallback(c);
    PyObject *comparison=PyObject_RichCompare(value,wanted,Py_NE);
    if (!comparison) return NULL;
    int unequal=PyObject_IsTrue(comparison); Py_DECREF(comparison);
    if (unequal<0) return NULL;
    if (kind==(PyObject *)&PyUnicode_Type && unequal) return fallback(c);
    if (unequal) return refuse(c);
    if (kind==(PyObject *)&PyFloat_Type && PyFloat_AS_DOUBLE(value)==0.0) {
        /* Rare observable math callbacks retain the original interpreter code,
           including its version-specific temporary destruction order. */
        PyObject *fn=lookup(c, PYTHON_LEAF);
        if (!fn) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,"missing original Python leaf"); return NULL; }
            PyObject *out=PyObject_CallFunctionObjArgs(fn,value,plan,NULL);
        Py_DECREF(fn);
        return out;
    }
    return Py_NewRef(wanted);
}
static PyObject *visit(Context *c, PyObject *value, PyObject *plan) {
    if (Py_EnterRecursiveCall(" in checked packed descriptor")<0) return NULL;
    PyObject *out=visit_body(c,value,plan);
    Py_LeaveRecursiveCall(); return out;
}
static PyObject *checked_copy(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3) {
        /* Preserve the existing public helper's argument-error wording. */
        PyObject *tuple=PyTuple_New(nargs), *a, *b, *d;
        if (!tuple) return NULL;
        for (Py_ssize_t i=0;i<nargs;i++) PyTuple_SET_ITEM(tuple,i,Py_NewRef(args[i]));
        (void)PyArg_ParseTuple(tuple,"OOO:copy",&a,&b,&d);
        Py_DECREF(tuple);
        return NULL;
    }
    PyObject *value=args[0],*plan=args[1],*globals=args[2];
    if (!PyDict_CheckExact(globals)) { PyErr_SetString(PyExc_TypeError,"exact globals required"); return NULL; }
    Context c={.globals=globals, .state=PyModule_GetState(self)}; return visit(&c,value,plan);
}
static PyMethodDef methods[]={{"copy",(PyCFunction)(void(*)(void))checked_copy,METH_FASTCALL,"Check every live descriptor value and copy containers."},{NULL,NULL,0,NULL}};
static int traverse(PyObject *module, visitproc visit, void *arg) {
    State *s=PyModule_GetState(module);
    for (int i=0;i<KEY_COUNT;i++) { Py_VISIT(s->keys[i]); }
    return 0;
}
static int clear(PyObject *module) {
    State *s=PyModule_GetState(module);
    for (int i=0;i<KEY_COUNT;i++) { Py_CLEAR(s->keys[i]); }
    return 0;
}
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_packed_native",NULL,sizeof(State),methods,NULL,traverse,clear,NULL};
PyMODINIT_FUNC PyInit__packed_native(void) {
    PyObject *out=PyModule_Create(&module); if (!out) return NULL;
    State *s=PyModule_GetState(out);
    for (int i=0;i<KEY_COUNT;i++) {
        s->keys[i]=PyUnicode_FromString(provider_names[i]);
        if (!s->keys[i]) { Py_DECREF(out); return NULL; }
    }
    return out;
}
