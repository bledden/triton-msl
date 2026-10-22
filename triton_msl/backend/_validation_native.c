/* Optional CPython implementations of ordered environment observation loops.
 * No unchecked cached 'unchanged' answer, private dict version or policy latch.
 * Generic attribute/call failures propagate; there is no retry after observation.
 * The Python callers retain selection, cache ownership, and native image checks.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
/* Proposed public-call gate, isolated for contract testing. No Metal calls.
 * A nonzero result is an extent decision, not allocator provenance authentication.
 */
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <unistd.h>
#include <malloc/malloc.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

typedef struct {
    size_t rounded, allocated;
    uintptr_t zone, default_zone;
    mach_vm_address_t region_start;
    mach_vm_size_t region_size;
    int region_status, protection;
    unsigned allocator_queries, vm_queries;
    const char *reason;
} extent_result;

static extent_result public_extent(const void *pointer, size_t length) {
    extent_result r = {0};
    uintptr_t ptr = (uintptr_t)pointer;
    long page_raw = sysconf(_SC_PAGESIZE);
    r.reason = "geometry";
    if (page_raw <= 0) return r;
    size_t page = (size_t)page_raw;
    if (!ptr || ptr % page || length < page || length % page == 0) return r;
    r.reason = "overflow";
    size_t extra = page - length % page;
    if (length > SIZE_MAX - extra) return r;
    size_t rounded = length + extra;
    if (ptr > UINTPTR_MAX - rounded) return r;
    r.allocator_queries++;
    r.allocated = malloc_size(pointer);
    r.reason = "no_malloc_extent";
    if (!r.allocated) return r;
    r.allocator_queries += 2;
    r.zone = (uintptr_t)malloc_zone_from_ptr(pointer);
    r.default_zone = (uintptr_t)malloc_default_zone();
    r.reason = "nondefault_zone";
    if (!r.zone || r.zone != r.default_zone) return r;
    r.reason = "short_extent";
    if (r.allocated < rounded) return r;
    mach_vm_address_t address = ptr;
    mach_vm_size_t size = 0;
    mach_port_t object = MACH_PORT_NULL;
    vm_region_basic_info_data_64_t info = {0};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    r.vm_queries++;
    r.region_status = mach_vm_region(mach_task_self(), &address, &size,
        VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    r.region_start = address; r.region_size = size; r.protection = info.protection;
    r.reason = "vm_region";
    if (r.region_status != KERN_SUCCESS || address > ptr || ptr - address > size ||
        rounded > size - (ptr - address) ||
        (info.protection & (VM_PROT_READ | VM_PROT_WRITE)) != (VM_PROT_READ | VM_PROT_WRITE)) return r;
    r.rounded = rounded; r.reason = "admitted";
    return r;
}

#if defined(Py_GIL_DISABLED)
#error "This optional native observation helper requires a GIL-enabled CPython build."
#endif
#if PY_VERSION_HEX < 0x030D0000
#error "This scratch implementation requires CPython 3.13+; older interpreters retain Python fallback."
#endif

/* Own interpreter-local names once. GetAttrString/GetItemString materialize
 * temporary unicode objects on every observation; that dominated the first
 * complete implementation, not the validation work itself. */
static const char *symbol_names[] = {
    "_native_builtins", "__builtins__", "getattr", "id", "tuple", "sys",
    "modules", "get", "__spec__", "origin", "submodule_search_locations",
    "__code__", "__defaults__", "__kwdefaults__", "__globals__", "all",
    "_native_all", "_dependency_checks", "function_state_is", "function_states_are"
};
enum {
    S_native_builtins, S_builtins, S_getattr, S_id, S_tuple, S_sys,
    S_modules, S_get, S_spec, S_origin, S_locations, S_code, S_defaults,
    S_kwdefaults, S_globals, S_all, S_native_all, S_dependency_checks,
    S_function_state_is, S_function_states_are
};
#define SYMBOL_COUNT (sizeof(symbol_names) / sizeof(symbol_names[0]))
#define KEY_CACHE_CAPACITY 128
#define PROBE_CACHE_CAPACITY 16
#define PROBE_CACHE_WIDTH 128
typedef struct {
    PyObject *probes;
    uint64_t epoch, serial;
    long kinds[PROBE_CACHE_WIDTH];
    Py_ssize_t mapping_count;
    PyObject *mappings[PROBE_CACHE_WIDTH * 2]; /* owned through probes */
    Py_ssize_t sizes[PROBE_CACHE_WIDTH * 2];
} ProbeCertificate;
enum { KC_EMPTY, KC_CERTIFIED, KC_DIRTY, KC_TOMBSTONE };
typedef struct { PyObject *dict; Py_ssize_t size; unsigned char state; } KeyCacheEntry;
typedef struct KeyCacheContext {
    PyInterpreterState *interp;
    int watcher_id, active, enabled;
    KeyCacheEntry entries[KEY_CACHE_CAPACITY];
    uint64_t censuses, hits, watch_events, watch_installs, table_full, registration_failures;
    uint64_t epoch;
    int epoch_enabled;
    struct KeyCacheContext *next;
} KeyCacheContext;
typedef struct {
    int host_extent_ready;
    const char *host_extent_selftest_reason;
    PyObject *symbols[SYMBOL_COUNT];
    KeyCacheContext *key_cache;
    uint64_t key_serial, probe_hits, probe_builds;
    int serial_exhausted;
    int collecting_programs;
    uint64_t released_programs;
    ProbeCertificate programs[PROBE_CACHE_CAPACITY];
} NativeState;
static KeyCacheContext *key_cache_contexts;

/* GIL-serialized raw bounded-table writes only: no allocation, refcount,
 * Python/C API call, or exception manipulation. */
static int key_cache_callback(PyDict_WatchEvent event, PyObject *dict,
                              PyObject *key, PyObject *new_value) {
    (void)key; (void)new_value;
    for (KeyCacheContext *ctx = key_cache_contexts; ctx; ctx = ctx->next) {
        if (!ctx->active) continue;
        /* Invalidate before the write. Never wrap into an old certificate.
         * Unrelated watched-dict writes conservatively invalidate too. */
        if (ctx->epoch == UINT64_MAX) ctx->epoch_enabled = 0;
        else ctx->epoch++;
        for (size_t i = 0; i < KEY_CACHE_CAPACITY; i++) {
            KeyCacheEntry *entry = &ctx->entries[i];
            if (entry->state != KC_EMPTY && entry->state != KC_TOMBSTONE && entry->dict == dict) {
                ctx->watch_events++;
                if (event == PyDict_EVENT_DEALLOCATED) {
                    entry->dict = NULL; entry->state = KC_TOMBSTONE;
                } else entry->state = KC_DIRTY;
            }
        }
    }
    return 0;
}

static inline PyObject *symbol(NativeState *st, int name) {
    return st->symbols[name];
}

/* All helpers below return owned references. Global lookup stays live across
 * callbacks, including Python replacements of normally-builtin operations. */
static PyObject *function_builtins(NativeState *st, PyObject *ns) {
    PyObject *value = PyDict_GetItemWithError(ns, symbol(st, S_native_builtins));
    if (!value) {
        if (PyErr_Occurred()) return NULL;
        value = PyDict_GetItemWithError(ns, symbol(st, S_builtins));
    }
    if (!value) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "missing function builtins mapping");
        return NULL;
    }
    if (PyModule_CheckExact(value)) value = PyModule_GetDict(value);
    if (!PyDict_CheckExact(value)) {
        PyErr_SetString(PyExc_TypeError, "native observation requires an exact function builtins dict");
        return NULL;
    }
    return Py_NewRef(value);
}

static PyObject *load_global(NativeState *st, PyObject *ns, PyObject *bn, int name) {
    PyObject *key = symbol(st, name);
    if (!key) return NULL;
    PyObject *value = PyDict_GetItemWithError(ns, key);
    if (PyErr_Occurred()) return NULL;
    if (value) return Py_NewRef(value);
    /* The MAPPING is captured, as in a Python function, not its live values. */
    value = PyDict_GetItemWithError(bn, key);
    if (PyErr_Occurred()) return NULL;
    if (value) return Py_NewRef(value);
    PyErr_Format(PyExc_NameError, "name '%s' is not defined", symbol_names[name]);
    return NULL;
}

static PyObject *get_attr(NativeState *st, PyObject *obj, int name) {
    if (!obj) return NULL;
    PyObject *key = symbol(st, name);
    PyObject *value = key ? PyObject_GetAttr(obj, key) : NULL;
    Py_DECREF(obj);
    return value;
}

#define global(ns, name) load_global(st, ns, bn, name)
#define attr(obj, name) get_attr(st, obj, name)

static PyObject *dict_get_owned(PyObject *mapping, PyObject *key, PyObject *fallback) {
    PyObject *value = PyDict_GetItemWithError(mapping, key);
    if (value) return Py_NewRef(value);
    return PyErr_Occurred() ? NULL : Py_NewRef(fallback);
}

/* This is the Python guard's complete observation sequence, including auditable
 * function.__code__/__defaults__ reads. No PyFunction_GET_CODE shortcut. */
static PyObject *implementation_unchanged(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    NativeState *st = PyModule_GetState(self);
    if (!st) return NULL;
    if (nargs != 2 || !PyDict_CheckExact(args[0])) {
        PyErr_SetString(PyExc_TypeError, "implementation_unchanged requires an exact globals dict and guard"); return NULL;
    }
    PyObject *ns = args[0], *guard = args[1];
    if (!PyTuple_CheckExact(guard) || PyTuple_GET_SIZE(guard) != 4) {
        PyErr_SetString(PyExc_TypeError, "implementation guard must be an owned four-tuple");
        return NULL;
    }
    PyObject *bn = function_builtins(st, ns);
    if (!bn) return NULL;
    PyObject *fn = PyTuple_GET_ITEM(guard, 0), *value = NULL, *deps = NULL;
    PyObject *iterator = NULL, *row = NULL, *globals_get = NULL, *builtins_get = NULL;
    PyObject *fallback = NULL, *found = NULL, *result = NULL, *all_fn = NULL;
    PyObject *globals_map = NULL, *builtins_map = NULL;
    const int attrs[] = {S_code, S_defaults, S_kwdefaults};
    PyObject *expected[] = {PyTuple_GET_ITEM(guard, 1), PyTuple_GET_ITEM(guard, 2), Py_None};
    for (int i = 0; i < 3; i++) {
        value = PyObject_GetAttr(fn, symbol(st, attrs[i]));
        if (!value) goto done;
        int same = value == expected[i];
        Py_CLEAR(value);
        if (!same) { result = Py_NewRef(Py_False); goto done; }
    }
    deps = PyTuple_GET_ITEM(guard, 3);
    all_fn = global(ns, S_all);
    if (!all_fn) goto done;
    if (all_fn != PyDict_GetItemWithError(ns, symbol(st, S_native_all))) {
        PyObject *make = global(ns, S_dependency_checks);
        if (!make) goto done;
        iterator = PyObject_GetIter(deps);
        if (iterator) value = PyObject_CallFunctionObjArgs(make, fn, iterator, NULL);
        Py_DECREF(make);
        if (value) result = PyObject_CallOneArg(all_fn, value);
        goto done;
    }
    iterator = PyObject_GetIter(deps);
    if (!iterator) goto done;
    while ((row = PyIter_Next(iterator))) {
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2) {
            PyErr_SetString(PyExc_TypeError, "implementation dependency must be a pair"); goto done;
        }
        PyObject *name = PyTuple_GET_ITEM(row, 0), *wanted = PyTuple_GET_ITEM(row, 1);
        globals_map = attr(Py_NewRef(fn), S_globals);
        if (!globals_map) goto done;
        if (!PyDict_CheckExact(globals_map)) {
            globals_get = attr(Py_NewRef(globals_map), S_get);
            if (!globals_get) goto done;
        }
        builtins_map = attr(Py_NewRef(fn), S_builtins);
        if (!builtins_map) goto done;
        if (!PyDict_CheckExact(builtins_map)) {
            builtins_get = attr(Py_NewRef(builtins_map), S_get);
            if (!builtins_get) goto done;
        }
        /* The fallback expression is evaluated FIRST, even if globals contains
         * the key. Exact builtin dict.get has no overridable method selection;
         * nonstandard mappings keep their already-observed callable. */
        fallback = builtins_get ? PyObject_CallOneArg(builtins_get, name)
                                : dict_get_owned(builtins_map, name, Py_None);
        if (!fallback) goto done;
        PyObject *arguments[] = {name, fallback};
        found = globals_get ? PyObject_Vectorcall(globals_get, arguments, 2, NULL)
                            : dict_get_owned(globals_map, name, fallback);
        if (!found) goto done;
        int same = found == wanted;
        Py_CLEAR(globals_get); Py_CLEAR(builtins_get); Py_CLEAR(fallback); Py_CLEAR(found); Py_CLEAR(row);
        Py_CLEAR(globals_map); Py_CLEAR(builtins_map);
        if (!same) { result = Py_NewRef(Py_False); goto done; }
    }
    if (!PyErr_Occurred()) result = Py_NewRef(Py_True);
done:
    Py_XDECREF(value); Py_XDECREF(iterator); Py_XDECREF(row); Py_XDECREF(globals_get);
    Py_XDECREF(builtins_get); Py_XDECREF(fallback); Py_XDECREF(found);
    Py_XDECREF(all_fn);
    Py_XDECREF(globals_map); Py_XDECREF(builtins_map);
    Py_DECREF(bn);
    return result;
}

/* Exact dictionaries only, insertion order and identity, no user equality. */
static PyObject *same_items(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2) { PyErr_SetString(PyExc_TypeError, "same_items requires two dictionaries"); return NULL; }
    PyObject *a = args[0], *b = args[1];
    if (!PyDict_CheckExact(a) || !PyDict_CheckExact(b)) Py_RETURN_FALSE;
    if (PyDict_Size(a) != PyDict_Size(b)) Py_RETURN_FALSE;
    Py_ssize_t pa = 0, pb = 0;
    PyObject *ka, *kb, *va, *vb;
    while (PyDict_Next(a, &pa, &ka, &va)) {
        if (!PyDict_Next(b, &pb, &kb, &vb) || ka != kb || va != vb) Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}


/* identity_probes(probes): evaluate an owned tuple of 5-tuples (kind, a, b, c, expected)
 * in order and answer True only if every probe holds. Probes compare LIVE objects
 * against identities saved when the derived stamp was computed; nothing is
 * materialized, cached, or retried. Any exception from an observation propagates
 * exactly as the equivalent Python getattr/dict access would raise it. */
enum {
    P_IDENT = 0,      /* a is expected */
    P_ATTR = 1,       /* getattr(a, b) is expected; AttributeError propagates */
    P_ATTRDEF = 2,    /* getattr(a, b, None) is expected; only AttributeError becomes None */
    P_EQATTRDEF = 3,  /* getattr(a, b, None) == expected */
    P_DICTGET = 4,    /* a[b] if present else c (absent sentinel); identity with expected; exact dict only */
    P_DEPGET = 5,     /* a.get(c, b.get(c)) identity with expected; a, b exact dicts */
    P_DICTITEMS = 6,  /* dict a has exactly keys b (list) and values c (list) by identity, in order */
    P_IDLIST = 7,     /* exact list a has the same length and element identities as tuple b */
    P_EQLIST = 8,     /* exact list a has the same length and equal elements as tuple b */
    P_CLOSURE0 = 9,   /* a.__closure__[0].cell_contents is expected */
    P_EQTUPLEATTRDEF = 10, /* tuple(getattr(a, b, None) or ()) == expected */
    P_TYPEOF = 11,    /* live Py_TYPE(a) is expected; never a cached type result */
    P_CLASSGET = 12,  /* own type dictionary, without invoking descriptors/metaclass hooks */
    P_TYPEVERSION = 13 /* nonzero CPython tag, invalidated through the entire MRO */
};

/* Cold-only schema reads: never execute a descriptor or metaclass callback. */
static PyObject *own_type_member(PyTypeObject *type, PyObject *name) {
    /* Builtin type dictionaries are interpreter-owned in 3.13/3.14; tp_dict
     * can be NULL. The public accessor returns a new reference. The type keeps
     * the dictionary/member alive after this temporary reference is released. */
    PyObject *dict = PyType_GetDict(type);
    if (!dict) return NULL;
    PyObject *found = PyDict_GetItemWithError(dict, name);
    Py_DECREF(dict);
    return found;
}

static PyObject *raw_type_member(PyTypeObject *type, PyObject *name) {
    if (!type->tp_mro || !PyTuple_CheckExact(type->tp_mro)) return NULL;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(type->tp_mro); i++) {
        PyTypeObject *base = (PyTypeObject *)PyTuple_GET_ITEM(type->tp_mro, i);
        PyObject *found = own_type_member(base, name);
        if (found || PyErr_Occurred()) return found;
    }
    return NULL;
}

static PyObject *mro_absent(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2 || !PyType_Check(args[0]) || !PyTuple_CheckExact(args[1])) {
        PyErr_SetString(PyExc_TypeError, "mro_absent requires a type and exact name tuple"); return NULL;
    }
    PyTypeObject *type = (PyTypeObject *)args[0];
    if (!type->tp_mro) Py_RETURN_FALSE;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args[1]); i++) {
        PyObject *name = PyTuple_GET_ITEM(args[1], i);
        if (!PyUnicode_CheckExact(name)) Py_RETURN_FALSE;
        PyObject *found = raw_type_member(type, name);
        if (PyErr_Occurred()) return NULL;
        if (found) Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

/* Verify the old class predicates before replacing them with a version guard.
 * Raw identities must match: classmethods/descriptors requiring binding decline.
 * Zero means uncacheable (including exhausted tags), never an unchanged answer.
 * Assign only here; the hot check must not refresh an invalidated saved tag. */
static PyObject *type_version(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2 || !PyType_Check(args[0]) || !PyTuple_CheckExact(args[1])) {
        PyErr_SetString(PyExc_TypeError, "type_version requires a type and owned checks"); return NULL;
    }
    PyTypeObject *type = (PyTypeObject *)args[0];
    if (Py_TYPE(type)->tp_getattro != PyType_Type.tp_getattro) return PyLong_FromLong(0);
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args[1]); i++) {
        PyObject *p = PyTuple_GET_ITEM(args[1], i);
        if (!PyTuple_CheckExact(p) || PyTuple_GET_SIZE(p) != 5 ||
            !PyLong_CheckExact(PyTuple_GET_ITEM(p, 0)) ||
            PyTuple_GET_ITEM(p, 1) != args[0] || !PyUnicode_CheckExact(PyTuple_GET_ITEM(p, 2))) {
            PyErr_SetString(PyExc_TypeError, "malformed type predicate"); return NULL;
        }
        long kind = PyLong_AsLong(PyTuple_GET_ITEM(p, 0));
        if (kind == -1 && PyErr_Occurred()) return NULL;
        PyObject *name = PyTuple_GET_ITEM(p, 2), *found;
        if (kind == P_CLASSGET) found = own_type_member(type, name);
        else if (kind == P_ATTR || kind == P_ATTRDEF) found = raw_type_member(type, name);
        else { PyErr_SetString(PyExc_TypeError, "unsupported type predicate"); return NULL; }
        if (PyErr_Occurred()) return NULL;
        if (!found) {
            if (kind == P_ATTR) return PyLong_FromLong(0);
            found = kind == P_CLASSGET ? PyTuple_GET_ITEM(p, 3) : Py_None;
        }
        if (found != PyTuple_GET_ITEM(p, 4)) return PyLong_FromLong(0);
    }
    if (!PyUnstable_Type_AssignVersionTag(type)) return PyLong_FromLong(0);
    return PyLong_FromUnsignedLong(type->tp_version_tag);
}

static PyObject *attr_default_none(PyObject *obj, PyObject *name) {
    PyObject *value = PyObject_GetAttr(obj, name);
    if (value) return value;
    if (PyErr_ExceptionMatches(PyExc_AttributeError)) { PyErr_Clear(); return Py_NewRef(Py_None); }
    return NULL;
}

/* A lookup in an exact dict can still call a stored str subclass's equality.
 * Certify every dict operand before the ordered probe pass begins.  The GIL
 * keeps other threads from inserting a key between this census and the pass;
 * the supported probes themselves are callback-free once all lookup keys are
 * exact strings.  Scan each shared dictionary only once by pointer identity. */
static int dict_keys_exact_strings_scan(PyObject *mapping) {
    if (!PyDict_CheckExact(mapping)) return 0;
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(mapping, &pos, &key, &value)) {
        if (!PyUnicode_CheckExact(key)) return 0;
    }
    return 1;
}

static int dict_keys_are_exact_strings(NativeState *st, PyObject *mapping) {
    if (!PyDict_CheckExact(mapping)) return 0;
    KeyCacheContext *ctx = st ? st->key_cache : NULL;
    if (!ctx || !ctx->active || !ctx->enabled || ctx->watcher_id < 0)
        return dict_keys_exact_strings_scan(mapping);
    KeyCacheEntry *free_entry = NULL;
    for (size_t i = 0; i < KEY_CACHE_CAPACITY; i++) {
        KeyCacheEntry *entry = &ctx->entries[i];
        if ((entry->state == KC_CERTIFIED || entry->state == KC_DIRTY) && entry->dict == mapping) {
            if (entry->state == KC_CERTIFIED && entry->size == PyDict_Size(mapping)) {
                ctx->hits++; return 1;
            }
            ctx->censuses++;
            if (!dict_keys_exact_strings_scan(mapping)) return 0;
            entry->size = PyDict_Size(mapping); entry->state = KC_CERTIFIED;
            return 1;
        }
        if (!free_entry && (entry->state == KC_EMPTY || entry->state == KC_TOMBSTONE)) free_entry = entry;
    }
    ctx->censuses++;
    if (!dict_keys_exact_strings_scan(mapping)) return 0;
    if (!free_entry) { ctx->table_full++; return 1; }
    if (PyDict_Watch(ctx->watcher_id, mapping) < 0) {
        PyErr_Clear(); ctx->enabled = 0; return 1;
    }
    free_entry->dict = mapping; free_entry->size = PyDict_Size(mapping);
    free_entry->state = KC_CERTIFIED; ctx->watch_installs++;
    return 1;
}

static PyObject *identity_probes(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    NativeState *st = PyModule_GetState(self);
    if (nargs != 1 || !PyTuple_CheckExact(args[0])) {
        PyErr_SetString(PyExc_TypeError, "identity_probes requires one owned tuple of probes"); return NULL;
    }
    PyObject *probes = args[0];
    Py_ssize_t count = PyTuple_GET_SIZE(probes);
    long decoded_kinds[PROBE_CACHE_WIDTH];
    int retain_kinds = count <= PROBE_CACHE_WIDTH;
    int certified = 0, may_certify = 0;
    uint64_t start_epoch = 0, start_serial = st->key_serial;
    PyObject *captured_mappings[PROBE_CACHE_WIDTH * 2];
    Py_ssize_t captured_sizes[PROBE_CACHE_WIDTH * 2], captured_count = 0;
    KeyCacheContext *ctx = st->key_cache;
    if (retain_kinds && ctx && ctx->active && ctx->enabled && ctx->epoch_enabled
            && st->key_serial) {
        for (size_t i = 0; i < PROBE_CACHE_CAPACITY; i++) {
            ProbeCertificate *p = &st->programs[i];
            if (p->probes == probes && p->serial == st->key_serial && p->epoch == ctx->epoch) {
                /* Watcher error reporting can re-enter before a pending write.
                 * Certify only the key domain, with the same live size guard as
                 * the existing key cache. Values are ALWAYS read below. A newly
                 * inserted foreign key changes size even if a pre-write hook
                 * refreshed this certificate after notification. */
                int same_sizes = 1;
                for (Py_ssize_t j = 0; j < p->mapping_count; j++) {
                    if (PyDict_Size(p->mappings[j]) != p->sizes[j]) { same_sizes = 0; break; }
                }
                if (!same_sizes) break;
                memcpy(decoded_kinds, p->kinds, (size_t)count * sizeof(long));
                start_epoch = ctx->epoch;
                certified = 1;
                st->probe_hits++;
                break;
            }
        }
    }
    if (certified) goto execute_probes;
    if (count > PY_SSIZE_T_MAX / 2 ||
        (size_t)count > SIZE_MAX / 2 / sizeof(PyObject *)) {
        PyErr_SetString(PyExc_OverflowError, "too many identity probes"); return NULL;
    }
    PyObject *local_mappings[32];
    PyObject **mappings = local_mappings;
    Py_ssize_t capacity = count * 2, mapping_count = 0;
    /* Deduplicate raw dictionary operands without repeatedly searching the
     * growing ordered list. The list still determines census order. This is
     * pointer hashing only: no Python hash/equality and no retained ownership.
     * Above the bounded table's load limit use the original linear search. */
    PyObject *seen_mappings[128] = {NULL};
    /* The preflight below already proves the exact immutable tuple/int schema
     * of EVERY probe before any dictionary lookup or callback. Retain the
     * decoded opcodes for the common-sized program instead of repeating those
     * schema tests and integer conversions in the ordered execution pass.
     * Larger public inputs keep the original conversion path: no new heap
     * allocation, input limit, or admission assumption is introduced. */
    /* This first pass performs no hashing, equality, attribute access, or user
     * iteration.  A foreign key declines the saved record without consuming
     * the callback; the unchanged evaluator then observes it in normal order. */
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *probe = PyTuple_GET_ITEM(probes, i);
        if (!PyTuple_CheckExact(probe) || PyTuple_GET_SIZE(probe) != 5 ||
            !PyLong_CheckExact(PyTuple_GET_ITEM(probe, 0))) {
            PyErr_SetString(PyExc_TypeError, "malformed identity probe");
            if (mappings != local_mappings) PyMem_Free(mappings);
            return NULL;
        }
        long kind = PyLong_AsLong(PyTuple_GET_ITEM(probe, 0));
        if (kind == -1 && PyErr_Occurred()) {
            if (mappings != local_mappings) PyMem_Free(mappings);
            return NULL;
        }
        if (retain_kinds) decoded_kinds[i] = kind;
        PyObject *operands[2];
        int n = 0;
        if (kind == P_DICTGET) {
            PyObject *query = PyTuple_GET_ITEM(probe, 2);
            if (!PyUnicode_CheckExact(query) && !PyBytes_CheckExact(query)) {
                if (mappings != local_mappings) PyMem_Free(mappings);
                Py_RETURN_FALSE;
            }
            operands[n++] = PyTuple_GET_ITEM(probe, 1);
        }
        else if (kind == P_DEPGET) {
            if (!PyUnicode_CheckExact(PyTuple_GET_ITEM(probe, 3))) {
                if (mappings != local_mappings) PyMem_Free(mappings);
                Py_RETURN_FALSE;
            }
            operands[n++] = PyTuple_GET_ITEM(probe, 1);
            operands[n++] = PyTuple_GET_ITEM(probe, 2);
        }
        for (int j = 0; j < n; j++) {
            int fresh;
            if (mapping_count < 96) {
                size_t slot = (((uintptr_t)operands[j] >> 4) * (uintptr_t)2654435761u) & 127u;
                while (seen_mappings[slot] && seen_mappings[slot] != operands[j]) slot = (slot + 1) & 127u;
                fresh = seen_mappings[slot] == NULL;
                if (fresh) seen_mappings[slot] = operands[j];
            } else {
                Py_ssize_t k = 0;
                while (k < mapping_count && mappings[k] != operands[j]) k++;
                fresh = k == mapping_count;
            }
            if (fresh) {
                if (mapping_count == (Py_ssize_t)(sizeof(local_mappings) / sizeof(local_mappings[0]))) {
                    PyObject **allocated = PyMem_Malloc((size_t)capacity * sizeof(*allocated));
                    if (!allocated) return PyErr_NoMemory();
                    memcpy(allocated, local_mappings, sizeof(local_mappings));
                    mappings = allocated;
                }
                mappings[mapping_count++] = operands[j];
            }
        }
    }
    for (Py_ssize_t i = 0; i < mapping_count; i++) {
        if (!dict_keys_are_exact_strings(st, mappings[i])) {
            if (mappings != local_mappings) PyMem_Free(mappings);
            Py_RETURN_FALSE;
        }
    }
    /* A successful census alone does not imply a watcher was installed: full
     * tables and exhausted watcher slots retain scanning. Certify only if EVERY
     * dict whose lookup we would skip is watched in this live context. */
    ctx = st->key_cache;
    may_certify = retain_kinds && ctx && ctx->active && ctx->enabled
                  && ctx->epoch_enabled && st->key_serial;
    for (Py_ssize_t i = 0; may_certify && i < mapping_count; i++) {
        int watched = 0;
        for (size_t j = 0; j < KEY_CACHE_CAPACITY; j++) {
            if (ctx->entries[j].dict == mappings[i] && ctx->entries[j].state == KC_CERTIFIED) {
                watched = 1;
                break;
            }
        }
        if (!watched) may_certify = 0;
    }
    if (may_certify) {
        start_epoch = ctx->epoch;
        start_serial = st->key_serial;
        captured_count = mapping_count;
        for (Py_ssize_t i = 0; i < mapping_count; i++) {
            captured_mappings[i] = mappings[i];
            captured_sizes[i] = PyDict_Size(mappings[i]);
        }
    }
    if (mappings != local_mappings) PyMem_Free(mappings);
execute_probes:
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *probe = PyTuple_GET_ITEM(probes, i);
        long kind = retain_kinds ? decoded_kinds[i] : PyLong_AsLong(PyTuple_GET_ITEM(probe, 0));
        if (kind == -1 && PyErr_Occurred()) return NULL;
        PyObject *a = PyTuple_GET_ITEM(probe, 1), *b = PyTuple_GET_ITEM(probe, 2);
        PyObject *c = PyTuple_GET_ITEM(probe, 3), *expected = PyTuple_GET_ITEM(probe, 4);
        PyObject *value = NULL;
        int ok;
        switch (kind) {
        case P_IDENT:
            if (a != expected) Py_RETURN_FALSE;
            break;
        case P_TYPEOF:
            if ((PyObject *)Py_TYPE(a) != expected) Py_RETURN_FALSE;
            break;
        case P_TYPEVERSION: {
            if (!PyType_Check(a) || !PyLong_CheckExact(expected)) Py_RETURN_FALSE;
            unsigned long tag = PyLong_AsUnsignedLong(expected);
            if (PyErr_Occurred()) return NULL;
            if (!tag || tag > UINT_MAX || ((PyTypeObject *)a)->tp_version_tag != tag) Py_RETURN_FALSE;
            break; }
        case P_CLASSGET:
            if (!PyType_Check(a) || !PyUnicode_CheckExact(b)) Py_RETURN_FALSE;
            value = own_type_member((PyTypeObject *)a, b);
            if (!value) { if (PyErr_Occurred()) return NULL; value = c; }
            if (value != expected) Py_RETURN_FALSE;
            break;
        case P_ATTR:
            value = PyObject_GetAttr(a, b); if (!value) return NULL;
            ok = (value == expected); Py_DECREF(value);
            if (!ok) Py_RETURN_FALSE;
            break;
        case P_ATTRDEF:
            value = attr_default_none(a, b); if (!value) return NULL;
            ok = (value == expected); Py_DECREF(value);
            if (!ok) Py_RETURN_FALSE;
            break;
        case P_EQATTRDEF:
            value = attr_default_none(a, b); if (!value) return NULL;
            if ((value != Py_None && !PyUnicode_CheckExact(value)) ||
                (expected != Py_None && !PyUnicode_CheckExact(expected))) {
                Py_DECREF(value); Py_RETURN_FALSE;
            }
            ok = PyObject_RichCompareBool(value, expected, Py_EQ); Py_DECREF(value);
            if (ok < 0) return NULL;
            if (!ok) Py_RETURN_FALSE;
            break;
        case P_DICTGET:
            /* Values are not certified by pre-write watcher notifications. */
            if (!PyDict_CheckExact(a)) Py_RETURN_FALSE;
            value = PyDict_GetItemWithError(a, b);
            if (!value) { if (PyErr_Occurred()) return NULL; value = c; }
            if (value != expected) Py_RETURN_FALSE;
            break;
        case P_DEPGET:
            if (!PyDict_CheckExact(a) || !PyDict_CheckExact(b)) Py_RETURN_FALSE;
            value = PyDict_GetItemWithError(b, c);            /* the fallback is evaluated first, as in Python */
            if (!value && PyErr_Occurred()) return NULL;
            { PyObject *primary = PyDict_GetItemWithError(a, c);
              if (!primary && PyErr_Occurred()) return NULL;
              if (primary) value = primary; }
            if (!value) value = Py_None;
            if (value != expected) Py_RETURN_FALSE;
            break;
        case P_DICTITEMS: {
            if (!PyDict_CheckExact(a) || !PyList_CheckExact(b) || !PyList_CheckExact(c)) Py_RETURN_FALSE;
            Py_ssize_t n = PyList_GET_SIZE(b);
            if (PyDict_Size(a) != n || PyList_GET_SIZE(c) != n) Py_RETURN_FALSE;
            Py_ssize_t pos = 0, j = 0; PyObject *k, *v;
            while (PyDict_Next(a, &pos, &k, &v)) {
                if (j >= n || k != PyList_GET_ITEM(b, j) || v != PyList_GET_ITEM(c, j)) Py_RETURN_FALSE;
                j++;
            }
            if (j != n) Py_RETURN_FALSE;
            break; }
        case P_IDLIST: {
            if (!PyList_CheckExact(a) || !PyTuple_CheckExact(b) || PyList_GET_SIZE(a) != PyTuple_GET_SIZE(b)) Py_RETURN_FALSE;
            for (Py_ssize_t j = 0; j < PyList_GET_SIZE(a); j++) if (PyList_GET_ITEM(a, j) != PyTuple_GET_ITEM(b, j)) Py_RETURN_FALSE;
            break; }
        case P_EQLIST: {
            if (!PyList_CheckExact(a) || !PyTuple_CheckExact(b) || PyList_GET_SIZE(a) != PyTuple_GET_SIZE(b)) Py_RETURN_FALSE;
            /* This speculative path is for ordinary sys.path strings only.
             * A user __eq__ belongs to the complete evaluator; besides replay,
             * it could resize a while we index the fixed saved tuple b. */
            for (Py_ssize_t j = 0; j < PyList_GET_SIZE(a); j++) {
                if (!PyUnicode_CheckExact(PyList_GET_ITEM(a, j)) ||
                    !PyUnicode_CheckExact(PyTuple_GET_ITEM(b, j))) Py_RETURN_FALSE;
            }
            for (Py_ssize_t j = 0; j < PyList_GET_SIZE(a); j++) {
                ok = PyObject_RichCompareBool(PyList_GET_ITEM(a, j), PyTuple_GET_ITEM(b, j), Py_EQ);
                if (ok < 0) return NULL;
                if (!ok) Py_RETURN_FALSE;
            }
            break; }
        case P_CLOSURE0: {
            value = PyObject_GetAttr(a, b); if (!value) return NULL;      /* b is "__closure__" */
            if (!PyTuple_CheckExact(value) || PyTuple_GET_SIZE(value) != 1 || !PyCell_Check(PyTuple_GET_ITEM(value, 0))) {
                Py_DECREF(value); Py_RETURN_FALSE;
            }
            PyObject *contents = PyCell_Get(PyTuple_GET_ITEM(value, 0));  /* new reference or NULL for an empty cell */
            Py_DECREF(value);
            ok = (contents == expected); Py_XDECREF(contents);
            if (!ok) Py_RETURN_FALSE;
            break; }
        case P_EQTUPLEATTRDEF: {
            value = attr_default_none(a, b); if (!value) return NULL;
            if (!PyTuple_CheckExact(expected) || (value != Py_None &&
                !PyList_CheckExact(value) && !PyTuple_CheckExact(value))) {
                Py_DECREF(value); Py_RETURN_FALSE;
            }
            Py_ssize_t n = value == Py_None ? 0 : PySequence_Size(value);
            if (n != PyTuple_GET_SIZE(expected)) { Py_DECREF(value); Py_RETURN_FALSE; }
            /* Validate every element before equality; no user bool/iter/eq. */
            for (Py_ssize_t j = 0; j < n; j++) {
                PyObject *item = PyList_CheckExact(value) ? PyList_GET_ITEM(value, j) : PyTuple_GET_ITEM(value, j);
                if (!PyUnicode_CheckExact(item) || !PyUnicode_CheckExact(PyTuple_GET_ITEM(expected, j))) {
                    Py_DECREF(value); Py_RETURN_FALSE;
                }
            }
            PyObject *materialized = value == Py_None ? PyTuple_New(0) : PySequence_Tuple(value);
            Py_DECREF(value);
            if (!materialized) return NULL;
            ok = PyObject_RichCompareBool(materialized, expected, Py_EQ); Py_DECREF(materialized);
            if (ok < 0) return NULL;
            if (!ok) Py_RETURN_FALSE;
            break; }
        default:
            PyErr_SetString(PyExc_TypeError, "unknown identity probe kind"); return NULL;
        }
    }
    ctx = st->key_cache;
    if (may_certify && st->key_serial == start_serial && ctx && ctx->active
            && ctx->enabled && ctx->epoch_enabled && ctx->epoch == start_epoch) {
        for (size_t i = 0; i < PROBE_CACHE_CAPACITY; i++) {
            ProbeCertificate *p = &st->programs[i];
            if (!p->probes || p->probes == probes) {
                /* No eviction on the launch path: releasing arbitrary owned
                 * objects could run finalizers. Capacity exhaustion just scans. */
                if (!p->probes) p->probes = Py_NewRef(probes);
                memcpy(p->kinds, decoded_kinds, (size_t)count * sizeof(long));
                p->mapping_count = captured_count;
                memcpy(p->mappings, captured_mappings, (size_t)captured_count * sizeof(PyObject *));
                memcpy(p->sizes, captured_sizes, (size_t)captured_count * sizeof(Py_ssize_t));
                p->serial = start_serial;
                p->epoch = start_epoch;
                st->probe_builds++;
                break;
            }
        }
    }
    Py_RETURN_TRUE;
}

static int key_cache_teardown(NativeState *st) {
    KeyCacheContext *ctx = st ? st->key_cache : NULL;
    if (!ctx) return 0;
    ctx->active = 0;
    for (size_t i = 0; i < KEY_CACHE_CAPACITY; i++) {
        KeyCacheEntry *entry = &ctx->entries[i];
        if ((entry->state == KC_CERTIFIED || entry->state == KC_DIRTY) && entry->dict) {
            if (PyDict_Unwatch(ctx->watcher_id, entry->dict) < 0) return -1;
            entry->dict = NULL; entry->state = KC_TOMBSTONE;
        }
    }
    if (ctx->watcher_id >= 0 && PyDict_ClearWatcher(ctx->watcher_id) < 0) return -1;
    KeyCacheContext **link = &key_cache_contexts;
    while (*link && *link != ctx) link = &(*link)->next;
    if (*link == ctx) *link = ctx->next;
    PyMem_Free(ctx); st->key_cache = NULL;
    return 0;
}

static int key_cache_init(NativeState *st) {
    KeyCacheContext *ctx = PyMem_Calloc(1, sizeof(*ctx));
    if (!ctx) { PyErr_NoMemory(); return -1; }
    ctx->interp = PyInterpreterState_Get();
    ctx->watcher_id = PyDict_AddWatcher(key_cache_callback);
    ctx->enabled = ctx->watcher_id >= 0;
    if (st->key_serial == UINT64_MAX) { st->serial_exhausted = 1; st->key_serial = 0; }
    else if (!st->serial_exhausted) st->key_serial++;
    ctx->epoch_enabled = st->key_serial != 0;
    ctx->active = 1;
    if (ctx->watcher_id < 0) { ctx->registration_failures++; PyErr_Clear(); }
    ctx->next = key_cache_contexts;
    key_cache_contexts = ctx;
    st->key_cache = ctx;
    return 0;
}

static PyObject *key_cache_reset(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    NativeState *st = PyModule_GetState(self);
    if (!st) return NULL;
    /* Detach ownership without running finalizers. Publish the replacement
     * context BEFORE releasing old probe programs; an object's __del__ may
     * reset recursively. There must be no outer context write after that. */
    PyObject *retired[PROBE_CACHE_CAPACITY];
    for (size_t i = 0; i < PROBE_CACHE_CAPACITY; i++) {
        retired[i] = st->programs[i].probes;
        st->programs[i].probes = NULL;
    }
    int rc = key_cache_teardown(st);
    if (rc == 0) rc = key_cache_init(st);
    PyObject *error = rc < 0 ? PyErr_GetRaisedException() : NULL;
    for (size_t i = 0; i < PROBE_CACHE_CAPACITY; i++) Py_XDECREF(retired[i]);
    if (error) { PyErr_SetRaisedException(error); return NULL; }
    Py_RETURN_NONE;
}

/* Explicit cold context only. A cache-owned program with refcount one cannot
 * be presented again by Python. Detach ALL such slots before any decref:
 * finalizers may call identity_probes(), collect recursively, or reset the
 * watcher context. Nothing below may overwrite a slot after callbacks begin.
 * Held programs (including the currently executing argument) are not evicted.
 * Cyclic/externally held programs conservatively retain their bounded slots.
 */
static PyObject *probe_cache_collect(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    NativeState *st = PyModule_GetState(self);
    if (st->collecting_programs) Py_RETURN_NONE;
    PyObject *retired[PROBE_CACHE_CAPACITY];
    size_t count = 0;
    st->collecting_programs = 1;
    for (size_t i = 0; i < PROBE_CACHE_CAPACITY; i++) {
        ProbeCertificate *p = &st->programs[i];
        if (p->probes && Py_REFCNT(p->probes) == 1) {
            retired[count++] = p->probes;
            p->probes = NULL;
        }
    }
    st->released_programs += count;
    for (size_t i = 0; i < count; i++) Py_DECREF(retired[i]);
    st->collecting_programs = 0;
    Py_RETURN_NONE;
}

static PyObject *key_cache_info(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    NativeState *st = PyModule_GetState(self);
    KeyCacheContext *ctx = st ? st->key_cache : NULL;
    size_t entries = 0;
    if (ctx) for (size_t i = 0; i < KEY_CACHE_CAPACITY; i++)
        entries += ctx->entries[i].state == KC_CERTIFIED || ctx->entries[i].state == KC_DIRTY;
    return Py_BuildValue("{s:i,s:i,s:n,s:K,s:K,s:K,s:K,s:K,s:K,s:K}",
        "active", ctx ? ctx->active : 0, "enabled", ctx ? ctx->enabled : 0,
        "entries", (Py_ssize_t)entries, "capacity", (unsigned long long)KEY_CACHE_CAPACITY,
        "censuses", (unsigned long long)(ctx ? ctx->censuses : 0),
        "hits", (unsigned long long)(ctx ? ctx->hits : 0),
        "watch_events", (unsigned long long)(ctx ? ctx->watch_events : 0),
        "watch_installs", (unsigned long long)(ctx ? ctx->watch_installs : 0),
        "table_full", (unsigned long long)(ctx ? ctx->table_full : 0),
        "registration_failures", (unsigned long long)(ctx ? ctx->registration_failures : 0));
}

static PyObject *probe_cache_info(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    NativeState *st = PyModule_GetState(self);
    size_t entries = 0;
    for (size_t i = 0; i < PROBE_CACHE_CAPACITY; i++) entries += st->programs[i].probes != NULL;
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K}", "entries", (unsigned long long)entries,
        "capacity", (unsigned long long)PROBE_CACHE_CAPACITY,
        "hits", (unsigned long long)st->probe_hits, "builds", (unsigned long long)st->probe_builds,
        "released", (unsigned long long)st->released_programs);
}

static PyObject *key_cache_exception_preserved(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    NativeState *st = PyModule_GetState(self);
    PyObject *watched = PyDict_New();
    if (!watched) return NULL;
    if (!dict_keys_are_exact_strings(st, watched)) { Py_DECREF(watched); Py_RETURN_FALSE; }
    uint64_t before = st->key_cache ? st->key_cache->watch_events : 0;
    PyObject *marker = PyObject_CallNoArgs(PyExc_RuntimeError);
    if (!marker) { Py_DECREF(watched); return NULL; }
    PyObject *keep = Py_NewRef(marker);
    PyErr_SetRaisedException(marker);  /* steals marker */
    int rc = key_cache_callback(PyDict_EVENT_MODIFIED, watched, NULL, NULL);
    PyObject *raised = PyErr_GetRaisedException();
    int same = rc == 0 && raised == keep && st->key_cache &&
               st->key_cache->watch_events == before + 1;
    Py_XDECREF(raised); Py_DECREF(keep);
    Py_DECREF(watched);  /* DEALLOCATED must tombstone the raw-pointer entry. */
    return PyBool_FromLong(same);
}

static PyObject *key_cache_enabled(PyObject *self, PyObject *arg) {
    NativeState *st = PyModule_GetState(self);
    int enabled = PyObject_IsTrue(arg);
    if (enabled < 0) return NULL;
    if (st && st->key_cache) st->key_cache->enabled = enabled;
    Py_RETURN_NONE;
}

/* Borrowed dictionary entries stay live under the required GIL. No lookup,
 * hashing, equality, iterator protocol, or value observation is performed. */
static PyObject *dict_keys_exact_str(PyObject *self, PyObject *value) {
    (void)self;
    return PyBool_FromLong(dict_keys_are_exact_strings(PyModule_GetState(self), value));
}

static PyObject *dict_keys_exact_bytes(PyObject *self, PyObject *value) {
    (void)self;
    if (!PyDict_CheckExact(value)) Py_RETURN_FALSE;
    Py_ssize_t pos = 0;
    PyObject *key, *item;
    while (PyDict_Next(value, &pos, &key, &item)) {
        if (!PyBytes_CheckExact(key)) Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject *identity_fast_path_disabled(PyObject *self, PyObject *module) {
    (void)self;
    if (!PyModule_CheckExact(module)) Py_RETURN_NONE;
    PyObject *state = PyModule_GetDict(module);
    if (!dict_keys_are_exact_strings(PyModule_GetState(self), state)) Py_RETURN_NONE;
    PyObject *mapping = PyDict_GetItemString(state, "environ");
    PyObject *owner = PyDict_GetItemString(state, "_Environ");
    if (!mapping || !owner || !PyType_Check(owner) || Py_TYPE(mapping) != (PyTypeObject *)owner) Py_RETURN_NONE;
    PyObject *instance = PyObject_GenericGetDict(mapping, NULL);
    if (!instance) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) { PyErr_Clear(); Py_RETURN_NONE; }
        return NULL;
    }
    if (!dict_keys_are_exact_strings(PyModule_GetState(self), instance)) { Py_DECREF(instance); Py_RETURN_NONE; }
    PyObject *data = PyDict_GetItemString(instance, "_data");
    if (!data || !PyDict_CheckExact(data)) { Py_DECREF(instance); Py_RETURN_NONE; }
    Py_ssize_t pos = 0; PyObject *stored, *item;
    while (PyDict_Next(data, &pos, &stored, &item)) {
        if (!PyBytes_CheckExact(stored)) { Py_DECREF(instance); Py_RETURN_NONE; }
    }
    PyObject *key = PyBytes_FromString("TRITON_MSL_IDENTITY_FAST_PATH");
    if (!key) { Py_DECREF(instance); return NULL; }
    PyObject *value = PyDict_GetItemWithError(data, key);
    Py_DECREF(key); Py_DECREF(instance);
    if (!value) { if (PyErr_Occurred()) return NULL; Py_RETURN_FALSE; }
    if (!PyBytes_CheckExact(value)) Py_RETURN_NONE;
    return PyBool_FromLong(PyBytes_GET_SIZE(value) == 1 && PyBytes_AS_STRING(value)[0] == '0');
}

static PyObject *bytes_dict_items_are(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2 || !PyDict_CheckExact(args[0]) || !PyTuple_CheckExact(args[1])) {
        PyErr_SetString(PyExc_TypeError, "bytes_dict_items_are requires an exact dict and tuple");
        return NULL;
    }
    PyObject *mapping = args[0], *pairs = args[1];
    Py_ssize_t pos = 0; PyObject *key, *value;
    while (PyDict_Next(mapping, &pos, &key, &value)) if (!PyBytes_CheckExact(key)) Py_RETURN_FALSE;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(pairs); i++) {
        PyObject *pair = PyTuple_GET_ITEM(pairs, i);
        if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 2 ||
            !PyBytes_CheckExact(PyTuple_GET_ITEM(pair, 0))) {
            PyErr_SetString(PyExc_TypeError, "malformed bytes-dict probe"); return NULL;
        }
        key = PyTuple_GET_ITEM(pair, 0);
        value = PyDict_GetItemWithError(mapping, key);
        if (!value && PyErr_Occurred()) return NULL;
        if (!value) value = Py_None;
        PyObject *expected = PyTuple_GET_ITEM(pair, 1);
        if (expected == Py_None) { if (value != Py_None) Py_RETURN_FALSE; }
        else if (value != expected) Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

/* Public CPython function metadata access without the object.__getattr__ audit
 * event emitted by Python-level fn.__code__.  This proves only artificial fast
 * guard metadata; the complete environment evaluator retains its audited reads. */
static PyObject *function_code_is(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2 || !PyFunction_Check(args[0]) || !PyCode_Check(args[1])) Py_RETURN_FALSE;
    return PyBool_FromLong(PyFunction_GetCode(args[0]) == args[1]);
}

static PyObject *function_state(PyObject *self, PyObject *fn) {
    (void)self;
    if (!PyFunction_Check(fn)) Py_RETURN_NONE;
    PyObject *code = PyFunction_GetCode(fn);
    PyObject *defaults = PyFunction_GetDefaults(fn); if (!defaults) defaults = Py_None;
    PyObject *kwdefaults = PyFunction_GetKwDefaults(fn); if (!kwdefaults) kwdefaults = Py_None;
    PyObject *closure = PyFunction_GetClosure(fn); if (!closure) closure = Py_None;
    PyObject *contents = PyTuple_New(closure == Py_None ? 0 : PyTuple_GET_SIZE(closure));
    if (!contents) return NULL;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(contents); i++) {
        PyObject *item = PyCell_Get(PyTuple_GET_ITEM(closure, i));
        if (!item) {
            Py_DECREF(contents);
            if (PyErr_Occurred()) return NULL;
            Py_RETURN_NONE;
        }
        PyTuple_SET_ITEM(contents, i, item);
    }
    PyObject *result = PyTuple_Pack(5, code, defaults, kwdefaults, closure, contents);
    Py_DECREF(contents); return result;
}

static PyObject *function_state_is(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2 || !PyFunction_Check(args[0]) || !PyTuple_CheckExact(args[1]) ||
        PyTuple_GET_SIZE(args[1]) != 5) Py_RETURN_FALSE;
    PyObject *fn = args[0], *state = args[1];
    PyObject *defaults = PyFunction_GetDefaults(fn); if (!defaults) defaults = Py_None;
    PyObject *kwdefaults = PyFunction_GetKwDefaults(fn); if (!kwdefaults) kwdefaults = Py_None;
    PyObject *closure = PyFunction_GetClosure(fn); if (!closure) closure = Py_None;
    if (PyFunction_GetCode(fn) != PyTuple_GET_ITEM(state, 0) ||
        defaults != PyTuple_GET_ITEM(state, 1) || kwdefaults != PyTuple_GET_ITEM(state, 2) ||
        closure != PyTuple_GET_ITEM(state, 3)) Py_RETURN_FALSE;
    PyObject *contents = PyTuple_GET_ITEM(state, 4);
    if (!PyTuple_CheckExact(contents) || closure == Py_None ||
        PyTuple_GET_SIZE(contents) != PyTuple_GET_SIZE(closure)) {
        return PyBool_FromLong(closure == Py_None && PyTuple_CheckExact(contents) && PyTuple_GET_SIZE(contents) == 0);
    }
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(contents); i++) {
        PyObject *item = PyCell_Get(PyTuple_GET_ITEM(closure, i));
        if (!item) {
            if (PyErr_Occurred()) return NULL;
            Py_RETURN_FALSE;
        }
        int same = item == PyTuple_GET_ITEM(contents, i); Py_DECREF(item);
        if (!same) Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

/* Batch only the ordinary, callback-free state loop. A changed observer, module
 * owner, key domain, aggregator, or malformed owned sequence returns
 * NotImplemented BEFORE Python fallback observes it. The same original
 * per-function primitive still makes every comparison in order. */
static PyObject *function_states_are(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "function_states_are requires native owner, states and any");
        return NULL;
    }
    PyObject *owner = args[0], *states = args[1], *aggregator = args[2];
    if (owner != self || Py_TYPE(owner) != &PyModule_Type || !PyTuple_CheckExact(states)
            || Py_TYPE(aggregator) != &PyCFunction_Type
            || strcmp(((PyCFunctionObject *)aggregator)->m_ml->ml_name, "any") != 0
            || PyCFunction_GET_FLAGS(aggregator) != METH_O
            || PyCFunction_GET_SELF(aggregator) == NULL
            || !PyModule_CheckExact(PyCFunction_GET_SELF(aggregator))
            || PyModule_GetDict(PyCFunction_GET_SELF(aggregator)) != PyEval_GetBuiltins())
        Py_RETURN_NOTIMPLEMENTED;
    NativeState *st = PyModule_GetState(self);
    PyObject *mapping = PyModule_GetDict(self);
    if (!dict_keys_are_exact_strings(st, mapping)) Py_RETURN_NOTIMPLEMENTED;
    PyObject *single = PyDict_GetItemWithError(mapping, symbol(st, S_function_state_is));
    if (!single && PyErr_Occurred()) return NULL;
    PyObject *batch = PyDict_GetItemWithError(mapping, symbol(st, S_function_states_are));
    if (!batch && PyErr_Occurred()) return NULL;
    if (!single || !PyCFunction_Check(single) || PyCFunction_GET_SELF(single) != self
            || PyCFunction_GET_FUNCTION(single) != (PyCFunction)(void (*)(void))function_state_is
            || !batch || !PyCFunction_Check(batch) || PyCFunction_GET_SELF(batch) != self
            || PyCFunction_GET_FUNCTION(batch) != (PyCFunction)(void (*)(void))function_states_are)
        Py_RETURN_NOTIMPLEMENTED;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(states); i++) {
        PyObject *pair = PyTuple_GET_ITEM(states, i);
        if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 2) Py_RETURN_NOTIMPLEMENTED;
        PyObject *arguments[2] = {PyTuple_GET_ITEM(pair, 0), PyTuple_GET_ITEM(pair, 1)};
        PyObject *result = function_state_is(self, arguments, 2);
        if (result != Py_True) return result;  /* False or the original exception. */
        Py_DECREF(result);
    }
    Py_RETURN_TRUE;
}


/* Sample the block-base behavior at module load; never treat this as a proof for
 * every allocator version or pointer. No user storage is accessed by this test. */
static void host_extent_selftest(NativeState *st) {
    st->host_extent_ready = 0;
    st->host_extent_selftest_reason = "selftest_allocation";
    long raw = sysconf(_SC_PAGESIZE);
    if (raw != 16384) { st->host_extent_selftest_reason = "selftest_page_size"; return; }
    size_t page = (size_t)raw, length = 4 * page + 1024;
    void *p = NULL;
    if (posix_memalign(&p, page, length)) return;
    extent_result base = public_extent(p, length);
    st->host_extent_selftest_reason = "selftest_positive";
    if (base.rounded != 5 * page) { free(p); return; }
    st->host_extent_selftest_reason = "selftest_interior";
    for (size_t offset = page; offset <= 2 * page; offset += page) {
        extent_result interior = public_extent((char *)p + offset, page + 1024);
        if (interior.rounded || strcmp(interior.reason, "no_malloc_extent")) { free(p); return; }
    }
    free(p);
#ifdef TRITON_MSL_TEST_EXTENT_SELFTEST_FAILURE
    st->host_extent_selftest_reason = "selftest_forced_failure";
    return;
#endif
    st->host_extent_ready = 1;
    st->host_extent_selftest_reason = "ready";
}

static PyObject *host_extent(PyObject *self, PyObject *args) {
    unsigned long long ptr, length;
    if (!PyArg_ParseTuple(args, "KK:host_extent", &ptr, &length)) return NULL;
    NativeState *st = PyModule_GetState(self);
    if (!st->host_extent_ready) return Py_BuildValue("Ks", 0ULL, "unavailable");
    extent_result result = public_extent((const void *)(uintptr_t)ptr, (size_t)length);
    return Py_BuildValue("Ks", (unsigned long long)result.rounded, result.reason);
}

static PyMethodDef methods[] = {
    {"host_extent", host_extent, METH_VARARGS, "Per-call default-zone backing extent; (span, reason), zero declines."},
    {"key_cache_info", key_cache_info, METH_NOARGS, "Scratch exact-key certificate statistics."},
    {"key_cache_reset", key_cache_reset, METH_NOARGS, "Scratch teardown and re-register diagnostic."},
    {"key_cache_enabled", key_cache_enabled, METH_O, "Scratch diagnostic cache toggle; false uses original census."},
    {"key_cache_exception_preserved", key_cache_exception_preserved, METH_NOARGS, "Scratch pending-exception callback control."},
    {"dict_keys_exact_str", dict_keys_exact_str, METH_O, "Callback-free exact-dict, all-exact-str key predicate."},
    {"dict_keys_exact_bytes", dict_keys_exact_bytes, METH_O, "Callback-free exact-dict, all-exact-bytes key predicate."},
    {"identity_fast_path_disabled", identity_fast_path_disabled, METH_O, "Raw identity fast-path opt-out bootstrap."},
    {"bytes_dict_items_are", (PyCFunction)(void (*)(void))bytes_dict_items_are, METH_FASTCALL, "Callback-free exact-bytes dict identity probes."},
    {"function_code_is", (PyCFunction)(void (*)(void))function_code_is, METH_FASTCALL, "Callback-free public function-code identity predicate."},
    {"function_state", function_state, METH_O, "Callback-free public function metadata snapshot."},
    {"function_state_is", (PyCFunction)(void (*)(void))function_state_is, METH_FASTCALL, "Callback-free public function metadata identity predicate."},
    {"function_states_are", (PyCFunction)(void (*)(void))function_states_are, METH_FASTCALL, "Ordered state batch, or NotImplemented before observer fallback."},
    {"implementation_unchanged", (PyCFunction)(void (*)(void))implementation_unchanged, METH_FASTCALL, "Complete function implementation guard."},
    {"same_items", (PyCFunction)(void (*)(void))same_items, METH_FASTCALL, "Exact-dict insertion-order identity comparison."},
    {"identity_probes", (PyCFunction)(void (*)(void))identity_probes, METH_FASTCALL, "Ordered live-object identity probes against saved identities."},
    {"probe_cache_info", probe_cache_info, METH_NOARGS, "Owned probe-certificate counters."},
    {"probe_cache_collect", probe_cache_collect, METH_NOARGS, "Cold-only release of cache-exclusive probe programs."},
    {"type_version", (PyCFunction)(void (*)(void))type_version, METH_FASTCALL, "Cold verified type version; zero declines."},
    {"mro_absent", (PyCFunction)(void (*)(void))mro_absent, METH_FASTCALL, "Callback-free whole-MRO absence check."},
    {NULL, NULL, 0, NULL}
};
static int traverse(PyObject *module, visitproc visit, void *arg) {
    NativeState *st = PyModule_GetState(module);
    for (size_t i = 0; i < SYMBOL_COUNT; i++) Py_VISIT(st->symbols[i]);
    for (size_t i = 0; i < PROBE_CACHE_CAPACITY; i++) Py_VISIT(st->programs[i].probes);
    return 0;
}
static int clear(PyObject *module) {
    NativeState *st = PyModule_GetState(module);
    for (size_t i = 0; i < SYMBOL_COUNT; i++) Py_CLEAR(st->symbols[i]);
    for (size_t i = 0; i < PROBE_CACHE_CAPACITY; i++) Py_CLEAR(st->programs[i].probes);
    return 0;
}
static void module_free(void *module) {
    NativeState *st = PyModule_GetState((PyObject *)module);
    if (key_cache_teardown(st) < 0) PyErr_WriteUnraisable((PyObject *)module);
}
static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "_validation_native", NULL, sizeof(NativeState), methods, NULL, traverse, clear, module_free
};
PyMODINIT_FUNC PyInit__validation_native(void) {
    PyObject *module = PyModule_Create(&definition);
    if (!module) return NULL;
    NativeState *st = PyModule_GetState(module);
    host_extent_selftest(st);
    if (PyModule_AddObjectRef(module, "host_extent_ready", st->host_extent_ready ? Py_True : Py_False) < 0 ||
        PyModule_AddStringConstant(module, "host_extent_selftest_reason", st->host_extent_selftest_reason) < 0) {
        Py_DECREF(module); return NULL;
    }
    if (key_cache_init(st) < 0) { Py_DECREF(module); return NULL; }
    for (size_t i = 0; i < SYMBOL_COUNT; i++) {
        st->symbols[i] = PyUnicode_InternFromString(symbol_names[i]);
        if (!st->symbols[i]) { Py_DECREF(module); return NULL; }
    }
    if (PyModule_AddObjectRef(module, "map_type", (PyObject *)&PyMap_Type) < 0) {
        Py_DECREF(module); return NULL;
    }
    return module;
}
