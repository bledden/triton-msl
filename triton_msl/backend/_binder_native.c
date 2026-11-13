/* Narrow flat i32 binder. Admission performs no Python callbacks; rejection
 * returns NotImplemented before traversing an observable Python prefix. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
/* Version-bound CPython 3.14 helper: read the installed monitoring layout,
 * never a guessed prefix of an opaque struct. */
#define Py_BUILD_CORE
#include "internal/pycore_instruments.h"
#undef Py_BUILD_CORE
#include <stdint.h>
#include <string.h>

typedef struct {
    PyObject *dict, *proxy, *packing;
} Recipe;
typedef struct { PyTypeObject *type; unsigned int version; } TypeProof;
typedef struct {
    PyObject *globals, *builtins, *bind, *bind_code, *plan_type;
    PyObject *fallback, *fallback_code;
    PyObject *module, *bind_key, *plan_keyword;
    PyObject *keys[16], *expected[16], *i32, *pointer, *class_name;
    PyTypeObject *struct_type;
    unsigned int plan_version;
    unsigned long long hits, misses, errors;
    TypeProof types[16];
    PyObject *scalar_fn,*scalar_code,*scalar_g,*scalar_b;
    PyObject *scalar_keys[14],*scalar_values[14],*scalar_attrs[3];
    TypeProof scalar_owners[8];
    unsigned long long scalar_hits,scalar_misses;
    PyObject *key_census; /* Owned callback-free native key-domain helper. */
} State;
static const char *binding_names[] = {
    "type", "len", "isinstance", "getattr", "callable", "enumerate", "zip",
    "int", "bool", "float", "str", "tuple", "list", "dict", "MappingProxyType", "_BindingPlan"
};

static int exact_keys(State *s, PyObject *dict) {
    if (s->key_census && dict) {
        PyObject *out = PyCFunction_GET_FUNCTION(s->key_census)(PyCFunction_GET_SELF(s->key_census), dict);
        if (!out) return 0;
        int ok = out == Py_True;
        Py_DECREF(out); return ok;
    }
    if (!dict || !PyDict_CheckExact(dict)) return 0;
    Py_ssize_t pos=0; PyObject *key, *value;
    while (PyDict_Next(dict,&pos,&key,&value)) if (!PyUnicode_CheckExact(key)) return 0;
    return 1;
}
static PyObject *lookup(State *s, PyObject *key) {
    PyObject *v=PyDict_GetItemWithError(s->globals,key);
    return v ? v : PyDict_GetItemWithError(s->builtins,key);
}
static int ordinary(State *s) {
    if (!s->globals || !exact_keys(s,s->globals) || !exact_keys(s,s->builtins)) return 0;
    if (PyDict_GetItemString(s->globals,"_binder_native")!=s->module) return 0;
    if (!PyFunction_Check(s->bind) || PyFunction_GET_CODE(s->bind)!=s->bind_code) return 0;
    if (PyFunction_GET_CODE(s->fallback)!=s->fallback_code) return 0;
    if (PyDict_GetItemString(s->globals,"bind_arguments")!=s->bind) return 0;
    if (!s->plan_version || ((PyTypeObject *)s->plan_type)->tp_version_tag!=s->plan_version) return 0;
    for (int i=0;i<16;i++) if (lookup(s,s->keys[i])!=s->expected[i]) return 0;
    return 1;
}
static void recipe_free(PyObject *capsule) {
    Recipe *r=PyCapsule_GetPointer(capsule,"triton_msl.flat_i32_binder");
    if (!r) { PyErr_Clear(); return; }
    Py_DECREF(r->dict); Py_DECREF(r->proxy); Py_DECREF(r->packing); PyMem_Free(r);
}

static PyObject *configure(PyObject *module, PyObject *args) {
    State *s=PyModule_GetState(module); PyObject *fn,*plan,*fallback,*census=Py_None;
    if (!PyArg_ParseTuple(args,"OOO|O",&fn,&plan,&fallback,&census)) return NULL;
    if (s->globals || !PyFunction_Check(fn) || !PyFunction_Check(fallback) || !PyType_Check(plan) || !PyType_IsSubtype((PyTypeObject *)plan,&PyTuple_Type)) Py_RETURN_FALSE;
    PyObject *g=PyFunction_GET_GLOBALS(fn), *b=((PyFunctionObject *)fn)->func_builtins;
    if (!exact_keys(s,g) || !exact_keys(s,b) || PyFunction_GET_GLOBALS(fallback)!=g ||
        ((PyFunctionObject *)fallback)->func_builtins!=b) Py_RETURN_FALSE;
    /* Optional sharing, never a new requirement for native/partial installs.
     * No Python callable or mutable module attribute is executed by the hot path. */
    if (PyCFunction_Check(census) && PyCFunction_GET_FLAGS(census)==METH_O &&
        !strcmp(((PyCFunctionObject *)census)->m_ml->ml_name,"dict_keys_exact_str")) {
        PyObject *owner=PyCFunction_GET_SELF(census);
        if (owner && PyModule_CheckExact(owner)) {
            PyModuleDef *definition=PyModule_GetDef(owner);
            PyObject *entries=PyModule_GetDict(owner);
            if (definition && !strcmp(definition->m_name,"_validation_native") &&
                exact_keys(s,entries) && PyDict_GetItemString(entries,"dict_keys_exact_str")==census)
                s->key_census=Py_NewRef(census);
        }
    }
    s->globals=Py_NewRef(g); s->builtins=Py_NewRef(b);
    s->bind=Py_NewRef(fn); s->bind_code=Py_NewRef(PyFunction_GET_CODE(fn)); s->plan_type=Py_NewRef(plan);
    s->fallback=Py_NewRef(fallback); s->fallback_code=Py_NewRef(PyFunction_GET_CODE(fallback));
    PyCodeObject *code=(PyCodeObject *)s->fallback_code;
    if (PyTuple_GET_SIZE(code->co_names)!=1 ||
        PyUnicode_CompareWithASCIIString(PyTuple_GET_ITEM(code->co_names,0),"bind_arguments")) Py_RETURN_FALSE;
    s->module=Py_NewRef(module);
    /* Preserve the original LOAD_GLOBAL name object's identity on fallback. */
    s->bind_key=Py_NewRef(PyTuple_GET_ITEM(code->co_names,0));
    for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(code->co_consts);i++) {
        PyObject *item=PyTuple_GET_ITEM(code->co_consts,i);
        if (PyTuple_CheckExact(item) && PyTuple_GET_SIZE(item)==1 &&
            PyUnicode_CheckExact(PyTuple_GET_ITEM(item,0)) &&
            PyUnicode_CompareWithASCIIString(PyTuple_GET_ITEM(item,0),"_plan")==0) {
            s->plan_keyword=Py_NewRef(item); break;
        }
    }
    if (!s->plan_keyword) Py_RETURN_FALSE;
    if (!PyUnstable_Type_AssignVersionTag((PyTypeObject *)plan)) Py_RETURN_FALSE;
    s->plan_version=((PyTypeObject *)plan)->tp_version_tag;
    PyObject *types[]={ (PyObject *)&PyType_Type,NULL,NULL,NULL,NULL,(PyObject *)&PyEnum_Type,(PyObject *)&PyZip_Type,
        (PyObject *)&PyLong_Type,(PyObject *)&PyBool_Type,(PyObject *)&PyFloat_Type,
        (PyObject *)&PyUnicode_Type,(PyObject *)&PyTuple_Type,(PyObject *)&PyList_Type,
        (PyObject *)&PyDict_Type,(PyObject *)&PyDictProxy_Type,plan };
    PyObject *builtin_module=PyImport_AddModule("builtins");
    for (int i=0;i<16;i++) {
        s->keys[i]=PyUnicode_FromString(binding_names[i]); if (!s->keys[i]) return NULL;
        PyObject *v=lookup(s,s->keys[i]);
        if (!v) Py_RETURN_FALSE;
        if (types[i]) { if (v!=types[i]) Py_RETURN_FALSE; }
        else if (!PyCFunction_Check(v) || PyCFunction_GET_SELF(v)!=builtin_module ||
                 strcmp(((PyCFunctionObject *)v)->m_ml->ml_name,binding_names[i])) Py_RETURN_FALSE;
        s->expected[i]=Py_NewRef(v);
    }
    s->i32=PyUnicode_FromString("i32"); s->pointer=PyUnicode_FromString("data_ptr"); s->class_name=PyUnicode_FromString("__class__");
    PyObject *st=PyImport_ImportModule("_struct"); if (!st) return NULL;
    PyObject *type=PyObject_GetAttrString(st,"Struct"); Py_DECREF(st);
    if (!type) return NULL;
    if (!PyType_Check(type) || !( ((PyTypeObject *)type)->tp_flags & Py_TPFLAGS_IMMUTABLETYPE) ||
        strcmp(((PyTypeObject *)type)->tp_name,"_struct.Struct")) { Py_DECREF(type); Py_RETURN_FALSE; }
    s->struct_type=(PyTypeObject *)type;
    if (!s->i32 || !s->pointer || !s->class_name) return NULL;
    Py_RETURN_TRUE;
}

static PyObject *make_parts(PyObject *module, PyObject *dict) {
    State *s=PyModule_GetState(module);
    if (!ordinary(s) || !s->struct_type || !exact_keys(s,dict)) Py_RETURN_NONE;
    /* A mixed/non-i32 plan stays on the original Python implementation even
     * if a later live declaration changes. This changes eligibility only. */
    if (PyDict_GET_SIZE(dict)!=1) Py_RETURN_NONE;
    PyObject *packing=PyDict_GetItemWithError(dict,s->i32);
    if (!packing || !PyTuple_CheckExact(packing) || PyTuple_GET_SIZE(packing)!=2 || PyTuple_GET_ITEM(packing,0)!=Py_False) Py_RETURN_NONE;
    PyObject *pack=PyTuple_GET_ITEM(packing,1);
    if (!PyCFunction_Check(pack)) Py_RETURN_NONE;
    PyObject *self=PyCFunction_GET_SELF(pack);
    if (!self || Py_TYPE(self)!=s->struct_type) Py_RETURN_NONE;
    PyObject *struct_dict=PyType_GetDict(s->struct_type); if (!struct_dict) return NULL;
    PyObject *descriptor=PyDict_GetItemString(struct_dict,"pack"); Py_DECREF(struct_dict);
    if (!descriptor || Py_TYPE(descriptor)!=&PyMethodDescr_Type ||
        PyCFunction_GET_FUNCTION(pack)!=((PyMethodDescrObject *)descriptor)->d_method->ml_meth) Py_RETURN_NONE;
    PyObject *format=PyObject_GetAttrString(self,"format"); if (!format) return NULL;
    int good=PyUnicode_CheckExact(format) && PyUnicode_CompareWithASCIIString(format,"<i")==0;
    Py_DECREF(format); if (!good) Py_RETURN_NONE;
    Recipe *r=PyMem_Calloc(1,sizeof(*r)); if (!r) return PyErr_NoMemory();
    r->dict=Py_NewRef(dict); r->packing=Py_NewRef(packing); r->proxy=PyDictProxy_New(dict);
    if (!r->proxy) { Py_DECREF(r->dict); Py_DECREF(r->packing); PyMem_Free(r); return NULL; }
    PyObject *cap=PyCapsule_New(r,"triton_msl.flat_i32_binder",recipe_free);
    if (!cap) { Py_DECREF(r->dict); Py_DECREF(r->proxy); Py_DECREF(r->packing); PyMem_Free(r); return NULL; }
    PyObject *result=PyTuple_Pack(2,r->proxy,cap); Py_DECREF(cap); return result;
}

/* A generic attribute lookup of a standard C method descriptor produces a
 * callable bound method without running the pointer's data_ptr function. */
static int certify_type(State *s, PyTypeObject *type) {
    if (type->tp_getattro!=PyObject_GenericGetAttr) return 0;
    for (int i=0;i<16;i++) if (s->types[i].type==type && s->types[i].version && s->types[i].version==type->tp_version_tag) return 1;
    PyObject *mro=type->tp_mro,*method=NULL,*cls=NULL;
    if (!mro || !PyTuple_CheckExact(mro)) return 0;
    for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(mro);i++) {
        PyObject *entry=PyTuple_GET_ITEM(mro,i); if (!PyType_Check(entry)) return 0;
        PyObject *dict=PyType_GetDict((PyTypeObject *)entry);
        if (!exact_keys(s,dict)) { Py_XDECREF(dict); return 0; }
        if (!method) method=PyDict_GetItemWithError(dict,s->pointer);
        if (!cls) cls=PyDict_GetItemWithError(dict,s->class_name);
        Py_DECREF(dict);
    }
    if (!method || Py_TYPE(method)!=&PyMethodDescr_Type || !PyType_IsSubtype(type,PyDescr_TYPE(method))) return 0;
    PyObject *base_dict=PyType_GetDict(&PyBaseObject_Type); if (!base_dict) return 0;
    PyObject *ordinary_class=PyDict_GetItemWithError(base_dict,s->class_name); Py_DECREF(base_dict);
    if (cls!=ordinary_class) return 0;
    if (!PyUnstable_Type_AssignVersionTag(type)) return 0;
    for (int i=0;i<16;i++) if (!s->types[i].type || s->types[i].type==type) {
        if (!s->types[i].type) s->types[i].type=(PyTypeObject *)Py_NewRef((PyObject *)type);
        s->types[i].version=type->tp_version_tag; return 1;
    }
    return 0;
}
typedef struct { State *s; int rejected; } Visit;
static int visit_dict(PyObject *value, void *raw) {
    Visit *v=raw;
    /* Managed inline attribute values lack key names through this API: decline
     * those conservatively. Materialized exact dictionaries can be inspected. */
    if (!exact_keys(v->s,value)) { v->rejected=1; return 1; }
    PyObject *override=PyDict_GetItemWithError(value,v->s->pointer);
    if (override) { v->rejected=1; return 1; }
    return 0;
}
static int pointer_ok(State *s, PyObject *value) {
    if (value==Py_None) return 1;
    /* The Python binder refuses tuple values before pointer lookup, including
     * tuple subclasses exposing an otherwise ordinary C method descriptor. */
    if (PyTuple_Check(value)) return 0;
    PyTypeObject *type=Py_TYPE(value);
    if (!certify_type(s,type)) return 0;
    if (type->tp_flags & Py_TPFLAGS_MANAGED_DICT) {
        Visit visit={s,0}; (void)PyObject_VisitManagedDict(value,visit_dict,&visit);
        return !visit.rejected;
    }
    /* No raw offsets or guessed third-party object layouts. */
    return type->tp_dictoffset==0;
}

typedef struct {
    PyObject *types[128]; int kinds[128]; int32_t integers[128];
    Py_ssize_t n, count;
} Flat;
static int admit(State *s, PyObject *const *args, Flat *f) {
    PyObject *values=args[0],*names=args[1],*signature=args[2],*plan=args[3];
    if (Py_TYPE(plan)!=(PyTypeObject *)s->plan_type || PyTuple_GET_SIZE(plan)!=2 ||
        !PyTuple_CheckExact(values) || !PyList_CheckExact(names) || !PyDict_CheckExact(signature)) return 0;
    PyObject *cap=PyTuple_GET_ITEM(plan,1);
    if (!PyCapsule_IsValid(cap,"triton_msl.flat_i32_binder")) return 0;
    Recipe *r=PyCapsule_GetPointer(cap,"triton_msl.flat_i32_binder");
    if (PyTuple_GET_ITEM(plan,0)!=r->proxy || !exact_keys(s,r->dict) || PyDict_GetItemWithError(r->dict,s->i32)!=r->packing) return 0;
    if (!exact_keys(s,signature)) return 0;
    Py_ssize_t n=PyList_GET_SIZE(names), count=0;
    if (n>128 || PyTuple_GET_SIZE(values)!=n) return 0;
    PyObject **types=f->types; int *kinds=f->kinds; int32_t *integers=f->integers;
    /* All admission precedes output construction. No descriptor, scalar
     * conversion hook, Python binder, or callbackful dictionary lookup occurs. */
    for (Py_ssize_t i=0;i<n;i++) {
        PyObject *name=PyList_GET_ITEM(names,i); if (!PyUnicode_CheckExact(name)) return 0;
        PyObject *ty=PyDict_GetItemWithError(signature,name); if (!ty || !PyUnicode_CheckExact(ty)) return 0;
        types[i]=ty;
        if (PyUnicode_CompareWithASCIIString(ty,"constexpr")==0) { kinds[i]=0; continue; }
        PyObject *value=PyTuple_GET_ITEM(values,i);
        if (PyUnicode_GET_LENGTH(ty)>0 && PyUnicode_ReadChar(ty,0)=='*') {
            if (!pointer_ok(s,value)) return 0;
            kinds[i]=1;
        } else if (PyUnicode_CompareWithASCIIString(ty,"i32")==0 && (PyLong_CheckExact(value) || PyBool_Check(value))) {
            int overflow=0; long long number=PyLong_AsLongLongAndOverflow(value,&overflow);
            if (overflow || PyErr_Occurred()) { PyErr_Clear(); return 0; }
            if (number<INT32_MIN || number>INT32_MAX) return 0;
            integers[i]=(int32_t)number; kinds[i]=2;
        } else return 0;
        count++;
    }
    /* Structural declines above are callback-free. Only potentially admitted
     * calls pay the live global/builtin census and packer-state proof. */
    if (!ordinary(s)) return 0;
    /* Struct's type is immutable; its INSTANCE can still be reinitialized. */
    PyObject *packer=PyTuple_GET_ITEM(r->packing,1);
    PyObject *format=PyObject_GetAttrString(PyCFunction_GET_SELF(packer),"format");
    if (!format) return -1;
    int format_ok=PyUnicode_CheckExact(format) && PyUnicode_CompareWithASCIIString(format,"<i")==0;
    Py_DECREF(format); if (!format_ok) return 0;
    f->n=n; f->count=count;
    return 1;
}
static void release_types(Flat *f) {
    for (Py_ssize_t i=0;i<f->n;i++) Py_DECREF(f->types[i]);
}
static PyObject *bind_flat_impl(PyObject *module, PyObject *const *args, Py_ssize_t nargs) {
    State *s=PyModule_GetState(module);
    if (nargs!=4) { PyErr_SetString(PyExc_TypeError,"bind_flat requires args, names, signature and plan"); return NULL; }
    Flat f={0}; int ok=admit(s,args,&f);
    if (ok<0) return NULL;
    if (!ok) Py_RETURN_NOTIMPLEMENTED;
    /* Hold each exact type string through output construction. On the gated
     * GIL-enabled CPython 3.14 runtime, container allocation schedules cyclic GC
     * for an evaluator safe point; it does not execute callbacks inside this C
     * window. No API below executes Python code or releases the GIL. */
    for (Py_ssize_t i=0;i<f.n;i++) Py_INCREF(f.types[i]);
    PyObject *result=PyTuple_New(4);
    if (!result) { release_types(&f); return NULL; }
    for (int i=0;i<4;i++) {
        PyObject *list=PyList_New(f.count);
        if (!list) { Py_DECREF(result); release_types(&f); return NULL; }
        PyTuple_SET_ITEM(result,i,list);
    }
    PyObject *values=args[0]; Py_ssize_t at=0;
    /* Exact longs and bytes are untracked; ref copies retain original leaves.
     * There is no Python call, descriptor lookup, or GIL release in this loop. */
    for (Py_ssize_t i=0;i<f.n;i++) {
        if (!f.kinds[i]) continue;
        PyObject *origin=PyLong_FromSsize_t(i),*payload=NULL;
        if (f.kinds[i]==1) payload=Py_NewRef(Py_None);
        else { uint32_t bits=(uint32_t)f.integers[i]; char bytes[4]; for (int k=0;k<4;k++) bytes[k]=(char)(bits>>(8*k)); payload=PyBytes_FromStringAndSize(bytes,4); }
        if (!origin || !payload) { Py_XDECREF(origin); Py_XDECREF(payload); Py_DECREF(result); release_types(&f); return NULL; }
        PyList_SET_ITEM(PyTuple_GET_ITEM(result,0),at,Py_NewRef(PyTuple_GET_ITEM(values,i)));
        PyList_SET_ITEM(PyTuple_GET_ITEM(result,1),at,Py_NewRef(f.types[i]));
        PyList_SET_ITEM(PyTuple_GET_ITEM(result,2),at,origin);
        PyList_SET_ITEM(PyTuple_GET_ITEM(result,3),at,payload); at++;
    }
    release_types(&f);
    return result;
}
static PyObject *bind_flat(PyObject *module, PyObject *const *args, Py_ssize_t nargs) {
    State *s=PyModule_GetState(module);
    PyObject *result=bind_flat_impl(module,args,nargs);
    if (!result) s->errors++;
    else if (result==Py_NotImplemented) s->misses++;
    else s->hits++;
    return result;
}
static PyObject *statistics(PyObject *module, PyObject *unused) {
    State *s=PyModule_GetState(module);
    return Py_BuildValue("{s:K,s:K,s:K}","hits",s->hits,"misses",s->misses,"errors",s->errors);
}
static PyObject *bind_with_plan(PyObject *module, PyObject *const *args,
                                Py_ssize_t nargs, PyObject *kwnames) {
    State *s=PyModule_GetState(module);
    if (!s->fallback) { PyErr_SetString(PyExc_RuntimeError,"native binder is not configured"); return NULL; }
    if (nargs==4 && (!kwnames || PyTuple_GET_SIZE(kwnames)==0)) {
        PyObject *result=bind_flat(module,args,nargs);
        if (result!=Py_NotImplemented) return result;
        Py_DECREF(result);
    }
    if (nargs==4 && (!kwnames || PyTuple_GET_SIZE(kwnames)==0) &&
        PyFunction_GET_CODE(s->fallback)==s->fallback_code) {
        /* Exact four-position shim: LOAD_GLOBAL using its actual co_names key,
         * then CALL with three positions and _plan. Do not capture the callee.
         * Foreign-key callbacks are observed once; never retry via the shim
         * after this lookup has performed an observable operation. */
        PyObject *fn=NULL;
        int found=PyDict_GetItemRef(s->globals,s->bind_key,&fn);
        if (found==0) found=PyDict_GetItemRef(s->builtins,s->bind_key,&fn);
        if (found<0) return NULL;
        if (!found) {
            PyErr_Format(PyExc_NameError,"name '%U' is not defined",s->bind_key);
            PyObject *error=PyErr_GetRaisedException();
            if (PyObject_SetAttrString(error,"name",s->bind_key)<0) { Py_DECREF(error); return NULL; }
            PyErr_SetRaisedException(error); return NULL;
        }
        PyObject *result=PyObject_Vectorcall(fn,args,3,s->plan_keyword);
        Py_DECREF(fn); return result;
    }
    /* Changed code, keyword calls, and arity handling remain in Python. */
    return PyObject_Vectorcall(s->fallback,args,(size_t)nargs,kwnames);
}
/* Optional post-hook scalar proof. Only an affirmative ordinary result bypasses
 * Python; all unsupported/custom/changed inputs keep its complete traversal. */
static const char *scalar_names[] = {
    "range", "len", "hasattr", "enumerate", "isinstance", "tuple", "type",
    "float", "int", "bool", "str", "bytes", "Exception",
    "_COMPILE_SHADER_SAFE_SCALAR_SIGS"
};
static PyObject *scalar_configure(PyObject *module, PyObject *fn) {
    State *s=PyModule_GetState(module);
    if (s->scalar_fn || !PyFunction_Check(fn)) Py_RETURN_FALSE;
    PyObject *g=PyFunction_GET_GLOBALS(fn), *b=((PyFunctionObject *)fn)->func_builtins;
    if (!exact_keys(s,g) || !exact_keys(s,b)) Py_RETURN_FALSE;
    PyObject *builtins=PyImport_AddModule("builtins");
    PyObject *types[]={ (PyObject *)&PyRange_Type,NULL,NULL,(PyObject *)&PyEnum_Type,NULL,
        (PyObject *)&PyTuple_Type,(PyObject *)&PyType_Type,(PyObject *)&PyFloat_Type,
        (PyObject *)&PyLong_Type,(PyObject *)&PyBool_Type,(PyObject *)&PyUnicode_Type,
        (PyObject *)&PyBytes_Type,PyExc_Exception,NULL };
    /* Store only after the complete cold admission, so a decline can be retried
       without leaving a partially initialized callable. */
    PyObject *keys[14]={0}, *values[14]={0}, *attrs[3]={0}; int admitted=0;
    for (int i=0;i<14;i++) {
        keys[i]=PyUnicode_FromString(scalar_names[i]); if (!keys[i]) goto done;
        PyObject *v=PyDict_GetItemWithError(g,keys[i]);
        if (!v) v=PyDict_GetItemWithError(b,keys[i]);
        if (!v) goto done;
        if (types[i]) { if (v!=types[i]) goto done; }
        else if (i<13) {
            if (!PyCFunction_Check(v) || PyCFunction_GET_SELF(v)!=builtins ||
                strcmp(((PyCFunctionObject *)v)->m_ml->ml_name,scalar_names[i])) goto done;
        } else {
            if (!PyFrozenSet_CheckExact(v)) goto done;
            PyObject *iterator=PyObject_GetIter(v), *entry;
            if (!iterator) goto done;
            int all_strings=1;
            while ((entry=PyIter_Next(iterator))) {
                if (!PyUnicode_CheckExact(entry)) all_strings=0;
                Py_DECREF(entry);
            }
            Py_DECREF(iterator); if (PyErr_Occurred() || !all_strings) goto done;
        }
        values[i]=Py_NewRef(v);
    }
    const char *fields[]={"arg_names","signature","constexpr_indices"};
    for (int i=0;i<3;i++) { attrs[i]=PyUnicode_InternFromString(fields[i]); if (!attrs[i]) goto done; }
    s->scalar_g=Py_NewRef(g); s->scalar_b=Py_NewRef(b);
    s->scalar_fn=Py_NewRef(fn); s->scalar_code=Py_NewRef(PyFunction_GET_CODE(fn));
    for (int i=0;i<14;i++) { s->scalar_keys[i]=keys[i]; s->scalar_values[i]=values[i]; keys[i]=values[i]=NULL; }
    for (int i=0;i<3;i++) { s->scalar_attrs[i]=attrs[i]; attrs[i]=NULL; }
    admitted=1;
done:
    for (int i=0;i<3;i++) Py_XDECREF(attrs[i]);
    for (int i=0;i<14;i++) { Py_XDECREF(keys[i]); Py_XDECREF(values[i]); }
    if (PyErr_Occurred()) return NULL;
    return PyBool_FromLong(admitted);
}
static int scalar_globals_hold(State *s) {
    if (!s->scalar_fn || !s->scalar_attrs[2] ||
        !exact_keys(s,s->scalar_g) || !exact_keys(s,s->scalar_b) ||
        PyFunction_GET_CODE(s->scalar_fn)!=s->scalar_code ||
        PyDict_GetItemString(s->scalar_g,"_compile_shader_scalars_ok")!=s->scalar_fn) return 0;
    for (int i=0;i<14;i++) {
        PyObject *v=PyDict_GetItemWithError(s->scalar_g,s->scalar_keys[i]);
        if (!v) v=PyDict_GetItemWithError(s->scalar_b,s->scalar_keys[i]);
        if (v!=s->scalar_values[i]) return 0;
    }
    return 1;
}
static int scalar_owner_plain(State *s, PyTypeObject *type) {
    if (type->tp_getattro!=PyObject_GenericGetAttr) return 0;
    for (int i=0;i<8;i++) if (s->scalar_owners[i].type==type &&
        s->scalar_owners[i].version && s->scalar_owners[i].version==type->tp_version_tag) return 1;
    PyObject *mro=type->tp_mro;
    if (!mro || !PyTuple_CheckExact(mro)) return 0;
    for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(mro);i++) {
        PyObject *base=PyTuple_GET_ITEM(mro,i);
        if (!PyType_Check(base)) return 0;
        PyObject *dict=PyType_GetDict((PyTypeObject *)base);
        if (!exact_keys(s,dict)) { Py_XDECREF(dict); return 0; }
        int shadow=0;
        for (int j=0;j<3;j++) if (PyDict_GetItemWithError(dict,s->scalar_attrs[j])) shadow=1;
        Py_DECREF(dict); if (shadow) return 0;
    }
    if (!PyUnstable_Type_AssignVersionTag(type)) return 0;
    for (int i=0;i<8;i++) if (!s->scalar_owners[i].type || s->scalar_owners[i].type==type) {
        if (!s->scalar_owners[i].type) s->scalar_owners[i].type=(PyTypeObject *)Py_NewRef((PyObject *)type);
        s->scalar_owners[i].version=type->tp_version_tag; break;
    }
    return 1;
}
static int scalar_admit(State *s, PyObject *launcher, PyObject *args, PyObject *checked) {
    PyThreadState *thread=PyThreadState_Get();
    if (thread->c_tracefunc || thread->c_profilefunc) return 0;
    if (s->scalar_code) {
        _PyCoMonitoringData *monitor=((PyCodeObject *)s->scalar_code)->_co_monitoring;
        if (monitor) for (int i=0;i<_PY_MONITORING_LOCAL_EVENTS;i++)
            if (monitor->active_monitors.tools[i]) return 0;
    }
    if (!s->scalar_fn || !s->scalar_attrs[2] ||
        (!PyList_CheckExact(args) && !PyTuple_CheckExact(args)) ||
        !PyTuple_CheckExact(checked) || PyTuple_GET_SIZE(checked)!=4) return 0;
    if (!scalar_owner_plain(s,Py_TYPE(launcher))) return 0;
    /* Public generic dictionary API, no guessed third-party layout. GIL-enabled
       CPython3.14 defers cyclic GC to evaluator safe points as in bind_flat. */
    PyObject *dict=PyObject_GenericGetDict(launcher,NULL);
    if (!dict) { if (PyErr_ExceptionMatches(PyExc_AttributeError)) {PyErr_Clear();return 0;} return -1; }
    int ok=0;
    if (!exact_keys(s,dict)) goto done;
    PyObject *names=PyDict_GetItemWithError(dict,s->scalar_attrs[0]);
    PyObject *signature=PyDict_GetItemWithError(dict,s->scalar_attrs[1]);
    PyObject *constants=PyDict_GetItemWithError(dict,s->scalar_attrs[2]);
    if (!names || !signature || !constants || !PyList_CheckExact(names) ||
        !exact_keys(s,signature) || (!PySet_CheckExact(constants) && !PyFrozenSet_CheckExact(constants))) goto done;
    Py_ssize_t n=PyList_GET_SIZE(names),count=PySequence_Fast_GET_SIZE(args),at=0;
    if (n>128 || count>n) goto done;
    for (int j=0;j<4;j++) if (!PyList_CheckExact(PyTuple_GET_ITEM(checked,j))) goto done;
    /* Set iteration is over an exact built-in container. Exact integer members
       prevent hash/equality callbacks during each original position test. */
    PyObject *iterator=PyObject_GetIter(constants), *item;
    if (!iterator) {ok=-1;goto done;}
    int keys_plain=1;
    while ((item=PyIter_Next(iterator))) {
        if (!PyLong_CheckExact(item) && !PyBool_Check(item)) keys_plain=0;
        Py_DECREF(item);
    }
    Py_DECREF(iterator); if (PyErr_Occurred()) {ok=-1;goto done;}
    if (!keys_plain) goto done;
    Py_ssize_t indices[128];
    for (Py_ssize_t i=0;i<n;i++) {
        PyObject *key=PyLong_FromSsize_t(i); if (!key) {ok=-1;goto done;}
        int present=PySet_Contains(constants,key); Py_DECREF(key);
        if (present<0) {ok=-1;goto done;}
        if (!present) indices[at++]=i;
    }
    if (at<count) goto done;
    for (Py_ssize_t j=0;j<count;j++) {
        PyObject *value=PySequence_Fast_GET_ITEM(args,j);
        if (value!=Py_None && !PyLong_CheckExact(value) && !PyBool_Check(value) && pointer_ok(s,value)) continue;
        if (!PyLong_CheckExact(value) && !PyBool_Check(value)) goto done;
        PyObject *name=PyList_GET_ITEM(names,indices[j]); if (!PyUnicode_CheckExact(name)) goto done;
        PyObject *sig=PyDict_GetItemWithError(signature,name);
        if (!sig || !PyUnicode_CheckExact(sig) || PyUnicode_CompareWithASCIIString(sig,"i32")) goto done;
        int contained=PySet_Contains(s->scalar_values[13],sig);
        if (contained<0) {ok=-1;goto done;} if (!contained) goto done;
        int overflow=0; long long number=PyLong_AsLongLongAndOverflow(value,&overflow);
        if (overflow || PyErr_Occurred()) {PyErr_Clear();goto done;}
        if (number<INT32_MIN || number>INT32_MAX) goto done;
        for (int k=0;k<4;k++) if (PyList_GET_SIZE(PyTuple_GET_ITEM(checked,k))<=j) goto done;
        if (PyList_GET_ITEM(PyTuple_GET_ITEM(checked,0),j)!=value ||
            PyList_GET_ITEM(PyTuple_GET_ITEM(checked,1),j)!=sig) goto done;
        PyObject *origin=PyList_GET_ITEM(PyTuple_GET_ITEM(checked,2),j);
        if (!PyLong_CheckExact(origin) || PyLong_AsSsize_t(origin)!=indices[j]) {if(PyErr_Occurred())PyErr_Clear();goto done;}
        if (!PyBytes_CheckExact(PyList_GET_ITEM(PyTuple_GET_ITEM(checked,3),j))) goto done;
    }
    /* Current providers are checked last, after all non-observing structure.
       No Python callback or GIL release occurs in this whole admission window. */
    ok=scalar_globals_hold(s);
done:
    Py_DECREF(dict); return ok;
}
static PyObject *scalar_gate(PyObject *module, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs!=3) {PyErr_SetString(PyExc_TypeError,"scalar_gate requires launcher, arguments and checked record");return NULL;}
    State *s=PyModule_GetState(module); int ok=scalar_admit(s,args[0],args[1],args[2]);
    if (ok<0) return NULL;
    if (ok) {s->scalar_hits++;Py_RETURN_TRUE;}
    s->scalar_misses++;Py_RETURN_NONE;
}
static PyObject *scalar_statistics(PyObject *module, PyObject *unused) {
    State *s=PyModule_GetState(module);
    return Py_BuildValue("{s:K,s:K}","hits",s->scalar_hits,"misses",s->scalar_misses);
}

static PyMethodDef methods[]={
    {"configure_scalar",scalar_configure,METH_O,NULL},
    {"scalar_gate",(PyCFunction)(void(*)(void))scalar_gate,METH_FASTCALL,NULL},
    {"scalar_statistics",scalar_statistics,METH_NOARGS,NULL},
    {"configure",configure,METH_VARARGS,NULL}, {"make_parts",make_parts,METH_O,NULL},
    {"statistics",statistics,METH_NOARGS,NULL},
    {"bind_with_plan",(PyCFunction)(void(*)(void))bind_with_plan,METH_FASTCALL|METH_KEYWORDS,NULL},
    {"bind_flat",(PyCFunction)(void(*)(void))bind_flat,METH_FASTCALL,NULL}, {NULL,NULL,0,NULL}
};
static int traverse(PyObject *module,visitproc visit,void *arg) {
    State *s=PyModule_GetState(module); Py_VISIT(s->scalar_fn);Py_VISIT(s->scalar_code);Py_VISIT(s->scalar_g);Py_VISIT(s->scalar_b);
    for(int i=0;i<14;i++){Py_VISIT(s->scalar_keys[i]);Py_VISIT(s->scalar_values[i]);}
    for(int i=0;i<3;i++)Py_VISIT(s->scalar_attrs[i]);
    for(int i=0;i<8;i++)Py_VISIT(s->scalar_owners[i].type);
    Py_VISIT(s->globals);Py_VISIT(s->builtins);Py_VISIT(s->bind);Py_VISIT(s->bind_code);Py_VISIT(s->plan_type);
    Py_VISIT(s->fallback);Py_VISIT(s->fallback_code);
    Py_VISIT(s->module);Py_VISIT(s->bind_key);Py_VISIT(s->plan_keyword);
    Py_VISIT(s->key_census);Py_VISIT(s->struct_type);Py_VISIT(s->i32);Py_VISIT(s->pointer);Py_VISIT(s->class_name);
    for(int i=0;i<16;i++){Py_VISIT(s->keys[i]);Py_VISIT(s->expected[i]);Py_VISIT(s->types[i].type);}return 0;
}
static int clear(PyObject *module) {
    State *s=PyModule_GetState(module);Py_CLEAR(s->scalar_fn);Py_CLEAR(s->scalar_code);Py_CLEAR(s->scalar_g);Py_CLEAR(s->scalar_b);
    for(int i=0;i<14;i++){Py_CLEAR(s->scalar_keys[i]);Py_CLEAR(s->scalar_values[i]);}
    for(int i=0;i<3;i++)Py_CLEAR(s->scalar_attrs[i]);
    for(int i=0;i<8;i++)Py_CLEAR(s->scalar_owners[i].type);
    Py_CLEAR(s->globals);Py_CLEAR(s->builtins);Py_CLEAR(s->bind);Py_CLEAR(s->bind_code);Py_CLEAR(s->plan_type);
    Py_CLEAR(s->fallback);Py_CLEAR(s->fallback_code);
    Py_CLEAR(s->module);Py_CLEAR(s->bind_key);Py_CLEAR(s->plan_keyword);
    Py_CLEAR(s->key_census);Py_CLEAR(s->struct_type);Py_CLEAR(s->i32);Py_CLEAR(s->pointer);Py_CLEAR(s->class_name);
    for(int i=0;i<16;i++){Py_CLEAR(s->keys[i]);Py_CLEAR(s->expected[i]);Py_CLEAR(s->types[i].type);}return 0;
}
static struct PyModuleDef definition={PyModuleDef_HEAD_INIT,"_binder_native",NULL,sizeof(State),methods,NULL,traverse,clear,NULL};
PyMODINIT_FUNC PyInit__binder_native(void){return PyModule_Create(&definition);}
