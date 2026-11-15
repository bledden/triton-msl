#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Test-only helper for tests/test_host_extent_gpu.py; not part of the package.
 * Conforming custom NumPy allocator: page-aligned, default-zone-backed allocations,
 * with 128 allocator-owned tail bytes. The logical NumPy data remains exactly n bytes.
 * The handler is immutable; NumPy retains it per array even after the global reset.
 */
static size_t allocation_calls, free_calls, bad_tails;
static void *custom_malloc(void *ctx, size_t n) {
    (void)ctx;
    if (n > SIZE_MAX - 128) return NULL;
    void *p = NULL;
    if (posix_memalign(&p, (size_t)getpagesize(), n + 128)) return NULL;
    memset((char *)p + n, 0xa7, 128);
    allocation_calls++;
    return p;
}
static void custom_free(void *ctx, void *p, size_t n) {
    (void)ctx;
    if (!p) return;
    for (size_t i = 0; i < 128; ++i) if (((unsigned char *)p)[n+i] != 0xa7) {bad_tails++; break;}
    free_calls++; free(p);
}
/* NumPy supplies no old size to realloc. For this test handler, store old lengths
 * in a side record would add needless machinery; delegate through a header-free
 * prefix-copy implementation using a bounded explicit record of outstanding blocks.
 * The probe never resizes, but a conforming handler still needs a real implementation.
 */
typedef struct Record { void *p; size_t n; struct Record *next; } Record;
static Record *records;
static void *tracked_malloc(void *ctx, size_t n) {
    Record *r = malloc(sizeof(*r)); if (!r) return NULL;
    void *p = custom_malloc(ctx, n); if (!p) {free(r); return NULL;}
    *r = (Record){p,n,records}; records=r; return p;
}
static void *tracked_calloc(void *ctx, size_t n, size_t size) {
    if (size && n > SIZE_MAX / size) return NULL;
    void *p=tracked_malloc(ctx,n*size); if(p)memset(p,0,n*size); return p;
}
static void tracked_free(void *ctx, void *p, size_t n) {
    if (!p) return;
    Record **slot=&records; while(*slot && (*slot)->p!=p)slot=&(*slot)->next;
    if(!*slot)abort();
    Record *r=*slot; *slot=r->next;
    if(r->n!=n)bad_tails++;
    custom_free(ctx,p,r->n); free(r);
}
static void *tracked_realloc(void *ctx, void *p, size_t n) {
    if(!p)return tracked_malloc(ctx,n);
    Record *r=records; while(r && r->p!=p)r=r->next;
    if(!r)return NULL;
    size_t old=r->n; void *q=tracked_malloc(ctx,n); if(!q)return NULL;
    memcpy(q,p,old<n?old:n); tracked_free(ctx,p,old); return q;
}
static PyDataMem_Handler handler={"extent_custom_default_backing",1,
    {NULL,tracked_malloc,tracked_calloc,tracked_realloc,tracked_free}};

static PyObject *install(PyObject *self, PyObject *unused) {
    (void)self;(void)unused;
    PyObject *capsule=PyCapsule_New(&handler,"mem_handler",NULL); if(!capsule)return NULL;
    PyObject *old=PyDataMem_SetHandler(capsule); Py_DECREF(capsule);return old;
}
static PyObject *restore(PyObject *self, PyObject *old) {
    (void)self;return PyDataMem_SetHandler(old);
}
static PyObject *stats(PyObject *self, PyObject *unused) {
    (void)self;(void)unused;
    return Py_BuildValue("{sK,sK,sK}","allocations",(unsigned long long)allocation_calls,
        "frees",(unsigned long long)free_calls,"bad_tails",(unsigned long long)bad_tails);
}
static PyMethodDef methods[]={
    {"install",install,METH_NOARGS,NULL},{"restore",restore,METH_O,NULL},
    {"stats",stats,METH_NOARGS,NULL},{NULL,NULL,0,NULL}};
static struct PyModuleDef def={PyModuleDef_HEAD_INIT,"_public_extent_probe",NULL,-1,methods,NULL,NULL,NULL,NULL};
PyMODINIT_FUNC PyInit__public_extent_probe(void) {import_array();return PyModule_Create(&def);}
