/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2015-2020 NVIDIA Corporation
 * Written by CUDA-GDB team at NVIDIA <cudatools@nvidia.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#include "defs.h"
#include "arch-utils.h"
#include "command.h"
#include "ui-out.h"
#include "cli/cli-script.h"
#include "gdbcmd.h"
#include "progspace.h"
#include "objfiles.h"
#include "value.h"
#include "language.h"
#include "python.h"
#include "python-internal.h"
#include <dlfcn.h>

#if HAVE_LIBPYTHON2_4
#define CONSTCHAR char
#else
#define CONSTCHAR const char
#endif

#define STRINGIFY2(name) #name
#define STRINGIFY(name) STRINGIFY2(name)

bool python_initialized = false;
static void *libpython_handle = NULL;

/* Dynamic references to constants */
PyObject *gdbpy_None = NULL;
PyObject *gdbpy_True = NULL;
PyObject *gdbpy_Zero = NULL;
PyObject *gdbpy_NotImplemented = NULL;
PyTypeObject *gdbpy_FloatType = NULL;
PyTypeObject *gdbpy_BoolType = NULL;
PyTypeObject *gdbpy_IntType = NULL;
PyTypeObject *gdbpy_LongType = NULL;
PyTypeObject *gdbpy_SliceType = NULL;
PyTypeObject *gdbpy_StringType = NULL;
PyTypeObject *gdbpy_ListType = NULL;
PyTypeObject *gdbpy_TupleType = NULL;
PyTypeObject *gdbpy_UnicodeType = NULL;

/* Dynamic reference to exception types */
PyObject **pgdbpyExc_AttributeError = NULL;
PyObject **pgdbpyExc_IndexError = NULL;
PyObject **pgdbpyExc_IOError = NULL;
PyObject **pgdbpyExc_KeyError = NULL;
PyObject **pgdbpyExc_KeyboardInterrupt  = NULL;
PyObject **pgdbpyExc_MemoryError = NULL;
PyObject **pgdbpyExc_NotImplementedError  = NULL;
PyObject **pgdbpyExc_OverflowError  = NULL;
PyObject **pgdbpyExc_RuntimeError = NULL;
PyObject **pgdbpyExc_StopIteration = NULL;
PyObject **pgdbpyExc_SystemError = NULL;
PyObject **pgdbpyExc_TypeError = NULL;
PyObject **pgdbpyExc_ValueError = NULL;
PyObject **pgdbpyExc_NameError = NULL;

PyThreadState **pgdbpy_OSReadlineTState = NULL;

char * (**pgdbpyOS_ReadlineFunctionPointer) (FILE *, FILE *,
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 4
			const char *) = NULL;
#else
			char *) = NULL;
#endif

PyThreadState **gdbpy_ThreadState_Current = NULL;

/* Imported functions */
int (*gdbpy_Arg_UnpackTuple) (PyObject *, const char *, Py_ssize_t, Py_ssize_t, ...) = NULL;
PyObject * (*gdbpy_ErrFormat)(PyObject *, const char *, ...) = NULL;
PyObject * (*gdbpy_BuildValue) (const char *, ...) = NULL;
PyObject * (*gdbpy_PyObject_CallFunctionObjArgs) (PyObject *,...) = NULL;
PyObject * (*gdbpy_PyObject_CallMethodObjArgs) (PyObject *, PyObject *,...) = NULL;
PyObject * (*gdbpy_PyObject_CallMethod)(PyObject *o, const char *m, const char *format, ...) = NULL;
PyObject * (*gdbpy_PyErr_NewException) (const char *name, PyObject *base, PyObject *dict) = NULL;
PyObject * (*gdbpy_PyString_FromStringAndSize) (const char *name, Py_ssize_t size) = NULL;
PyObject * (*gdbpy_PyBytes_FromStringAndSize) (const char *name, Py_ssize_t size) = NULL;

#ifdef IS_PY3K
PyObject * (*gdbpy_PyMemoryView_FromObject) () = NULL;
int (*gdbpy_PySlice_GetIndicesEx) (PyObject *slice, Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step, Py_ssize_t *slicelength) = NULL;
#else
PyObject * (*gdbpy_PyBuffer_FromObject) (PyObject *base, Py_ssize_t offset, Py_ssize_t size) = NULL;
int (*gdbpy_PySlice_GetIndicesEx) (PySliceObject *slice, Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step, Py_ssize_t *slicelength) = NULL;
#endif

#ifdef HAVE_LONG_LONG

PyObject * (*gdbpy_PyLong_FromLongLong) (long long val) = NULL;
PyObject * (*gdbpy_PyLong_FromUnsignedLongLong) (unsigned long long val) = NULL;
unsigned long long (*gdbpy_PyLong_AsUnsignedLongLong) (PyObject *obj) = NULL;

#else /* HAVE_LONG_LONG */

PyObject * (*gdbpy_PyLong_FromLongLong) (long val) = NULL;
PyObject * (*gdbpy_PyLong_FromUnsignedLongLong) (unsigned long val) = NULL;
unsigned long (*gdbpy_PyLong_AsUnsignedLongLong) (PyObject *obj) = NULL;

#endif /* HAVE_LONG_LONG */

int (*gdbpy_PySlice_Check) (PyObject *) = NULL;

int (*gdbpy_PyArg_ParseTuple) (PyObject *obj, const char *, ...) = NULL;
int (*gdbpy_PyArg_ParseTupleAndKeywords) (PyObject *obj, PyObject *, const char *, char **, ...) = NULL;
int (*gdbpy_PyArg_VaParseTupleAndKeywords) (PyObject *obj, PyObject *, const char *, char **, ...);
PyObject * (*gdbpy_StringFromFormat) (const char *, ...) = NULL;
PyObject * (*gdbpy_PySequence_Concat) (PyObject *o1, PyObject *o2) = NULL;
PY_LONG_LONG (*gdbpy_Long_AsLong) (PyObject *) = NULL;
int (*gdbpy_Dict_SetItem) (PyObject *mp, PyObject *key, PyObject *item) = NULL;
PyObject * (*gdbpy_Dict_Keys) (PyObject *mp) = NULL;
void (*gdbpy_Err_SetNone) (PyObject *) = NULL;

int (*gdbpy_PyObject__IsInstance) (PyObject *object, PyObject *typeorclass) = NULL;
PyObject * (*gdbpy_PySys_GetObject) (const char *) = NULL;
void (*gdbpy_PySys_SetPath) (GDB_PYSYS_SETPATH_CHAR *path) = NULL;
Py_ssize_t (*gdbpy_PyInt_AsSize_t) (PyObject *) = NULL;
PyObject * (*gdbpy_PyInt_FromSize_t) (size_t size) = NULL;
PyObject * (*gdbpy_PyInt_FromLong) (long val) = NULL;
int (*gdbpy_PyObject_GetBuffer) (PyObject *o, Py_buffer *buf, int f) = NULL;
void (*gdbpy_PyBuffer_Release) (Py_buffer *buf) = NULL;

static PyObject * (*gdb_PyBool_FromLong) (long) = NULL;
static PyObject * (*gdb_PyBuffer_FromReadWriteObject) (PyObject *base, Py_ssize_t offset, Py_ssize_t size) = NULL;
static int (*gdb_PyCallable_Check) (PyObject *o) = NULL;
static PyObject * (*gdb_PyDict_New) (void) = NULL;
static int (*gdb_PyDict_SetItemString) (PyObject *dp, const char *key, PyObject *item) = NULL;
static void (*gdb_PyErr_Clear) (void) = NULL;
static int (*gdb_PyErr_ExceptionMatches) (PyObject *) = NULL;
static void (*gdb_PyErr_Fetch) (PyObject **, PyObject **, PyObject **) = NULL;
static int (*gdb_PyErr_GivenExceptionMatches) (PyObject *, PyObject *) = NULL;
static PyObject * (*gdb_PyErr_Occurred) (void) = NULL;
static void (*gdb_PyErr_Print) (void) = NULL;
static void (*gdb_PyErr_Restore) (PyObject *, PyObject *, PyObject *) = NULL;
static PyObject * (*gdb_PyErr_SetFromErrno) (PyObject *) = NULL;
static void (*gdb_PyErr_SetInterrupt) (void) = NULL;
static void (*gdb_PyErr_SetObject) (PyObject *, PyObject *) = NULL;
static void (*gdb_PyErr_SetString) (PyObject *, const char *) = NULL;
//static PyObject * (*gdb_PyErr_NewException)(const char *name, PyObject *base, PyObject *dict) = NULL;
static void (*gdb_PyEval_InitThreads) (void) = NULL;
static void (*gdb_PyEval_ReleaseLock) (void) = NULL;
static void (*gdb_PyEval_RestoreThread) (PyThreadState *) = NULL;
static PyThreadState * (*gdb_PyEval_SaveThread) (void) = NULL;
static double (*gdb_PyFloat_AsDouble) (PyObject *) = NULL;
static PyObject * (*gdb_PyFloat_FromDouble) (double) = NULL;
static PyFrameObject * (*gdb_PyFrame_New)(PyThreadState *, PyCodeObject *, PyObject *, PyObject *) = NULL;
static PyGILState_STATE (*gdb_PyGILState_Ensure) (void) = NULL;
static void (*gdb_PyGILState_Release) (PyGILState_STATE) = NULL;
static PyObject * (*gdb_PyImport_AddModule) (const char *name) = NULL;
static PyObject * (*gdb_PyImport_ImportModule) (const char *name) = NULL;
static long (*gdb_PyInt_AsLong) (PyObject *) = NULL;
static PyObject * (*gdb_PyInt_FromLong) (long) = NULL;
static long (*gdb_PyInt_GetMax) (void) = NULL;
static PyObject * (*gdb_PyIter_Next) (PyObject *) = NULL;
static int (*gdb_PyList_Append) (PyObject *, PyObject *) = NULL;
static PyObject * (*gdb_PyList_AsTuple) (PyObject *) = NULL;
static PyObject * (*gdb_PyList_GetItem) (PyObject *, Py_ssize_t) = NULL;
static int (*gdb_PyList_Insert) (PyObject *, Py_ssize_t, PyObject *) = NULL;
static PyObject * (*gdb_PyList_New) (Py_ssize_t size) = NULL;
static Py_ssize_t (*gdb_PyList_Size) (PyObject *) = NULL;
static PY_LONG_LONG (*gdb_PyLong_AsLongLong) (PyObject *) = NULL;
static unsigned PY_LONG_LONG (*gdb_PyLong_AsUnsignedLongLong) (PyObject *) = NULL;
static PyObject * (*gdb_PyLong_FromLong) (long) = NULL;
static PyObject * (*gdb_PyLong_FromLongLong) (PY_LONG_LONG) = NULL;
static PyObject * (*gdb_PyLong_FromUnsignedLong) (unsigned long) = NULL;
static PyObject * (*gdb_PyLong_FromUnsignedLongLong) (unsigned PY_LONG_LONG) = NULL;
static void * (*gdb_PyMem_Malloc) (size_t) = NULL;
static int (*gdb_PyModule_AddIntConstant) (PyObject *, const char *, long) = NULL;
static int (*gdb_PyModule_AddObject) (PyObject *, const char *, PyObject *) = NULL;
static int (*gdb_PyModule_AddStringConstant) (PyObject *, const char *, const char *) = NULL;
static PyObject * (*gdb_PyModule_GetDict) (PyObject *) = NULL;
static PyObject * (*gdb_PyNumber_Long) (PyObject *o) = NULL;
static int (*gdb_PyOS_InterruptOccurred) (void) = NULL;
static int (*gdb_PyObject_AsReadBuffer) (PyObject *obj, const void **, Py_ssize_t *) = NULL;
static int (*gdb_PyObject_CheckReadBuffer) (PyObject *obj) = NULL;
static PyObject * (*gdb_PyObject_GenericGetAttr) (PyObject *, PyObject *) = NULL;
static int (*gdb_PyObject_GenericSetAttr)(PyObject *arg1, PyObject *arg2, PyObject *arg3) = NULL;
static PyObject * (*gdb_PyObject_GetAttr) (PyObject *, PyObject *) = NULL;
static PyObject * (*gdb_PyObject_GetAttrString) (PyObject *, const char *) = NULL;
static PyObject * (*gdb_PyObject_GetIter) (PyObject *) = NULL;
static int (*gdb_PyObject_HasAttr) (PyObject *, PyObject *) = NULL;
static int (*gdb_PyObject_HasAttrString) (PyObject *, const char *) = NULL;
static int (*gdb_PyObject_IsTrue) (PyObject *) = NULL;
static int (*gdb_PyObject_RichCompareBool) (PyObject *, PyObject *, int) = NULL;
static int (*gdb_PyObject_SetAttrString) (PyObject *, const char *, PyObject *) = NULL;
static PyObject * (*gdb_PyObject_Str) (PyObject *) = NULL;
static int (*gdb_PyRun_InteractiveLoopFlags) (FILE *, const char *, PyCompilerFlags *) = NULL;
static int (*gdb_PyRun_SimpleFileExFlags) (FILE *, const char *, int, PyCompilerFlags *) = NULL;
static int (*gdb_PyRun_SimpleStringFlags) (const char *, PyCompilerFlags *) = NULL;
static PyObject * (*gdb_PyRun_StringFlags)(const char *, int, PyObject *, PyObject *, PyCompilerFlags *) = NULL;
static int (*gdb_PySequence_Check) (PyObject *o) = NULL;
static int (*gdb_PySequence_DelItem) (PyObject *o, Py_ssize_t i) = NULL;
static PyObject * (*gdb_PySequence_GetItem) (PyObject *o, Py_ssize_t i) = NULL;
static Py_ssize_t (*gdb_PySequence_Index) (PyObject *o, PyObject *value) = NULL;
static PyObject * (*gdb_PySequence_List) (PyObject *o) = NULL;
static Py_ssize_t (*gdb_PySequence_Size) (PyObject *o) = NULL;
static char * (*gdb_PyString_AsString) (PyObject *) = NULL;
static PyObject * (*gdb_PyString_Decode) (const char *, Py_ssize_t, const char *, const char *) = NULL;
static PyObject * (*gdb_PyString_FromString) (const char *) = NULL;
static Py_ssize_t (*gdb_PyString_Size) (PyObject *) = NULL;
//static PyObject * (*gdb_PySys_GetObject) (char *) = NULL;
//static void (*gdb_PySys_SetPath) (char *) = NULL;
static PyThreadState * (*gdb_PyThreadState_Get) (void) = NULL;
static PyThreadState * (*gdb_PyThreadState_Swap) (PyThreadState *) = NULL;
static PyObject * (*gdb_PyTuple_GetItem) (PyObject *, Py_ssize_t) = NULL;
static PyObject * (*gdb_PyTuple_New) (Py_ssize_t size) = NULL;
static int (*gdb_PyTuple_SetItem) (PyObject *, Py_ssize_t, PyObject *) = NULL;
static Py_ssize_t (*gdb_PyTuple_Size) (PyObject *) = NULL;
static PyObject * (*gdb_PyType_GenericNew)(PyTypeObject *, PyObject *, PyObject *) = NULL;
static int (*gdb_PyType_IsSubtype) (PyTypeObject *, PyTypeObject *) = NULL;
static int (*gdb_PyType_Ready) (PyTypeObject *) = NULL;
static void (*gdb_Py_Finalize) (void) = NULL;
static int (*gdb_Py_FlushLine) (void) = NULL;
static void (*gdb_Py_Initialize) (void) = NULL;
static PyObject * (*gdb_Py_InitModule4)(const char *, PyMethodDef *, const char *, PyObject *, int) = NULL;
static PyObject * (*gdb_Py_InitModule4_64)(const char *, PyMethodDef *, const char *, PyObject *, int) = NULL;
static void (*gdb_Py_SetProgramName) (char *) = NULL;
static PyObject * (*gdb__PyObject_New) (PyTypeObject *) = NULL;
static PyCodeObject * (*gdb_PyCode_New) (int, int, int, int,
           PyObject *, PyObject *, PyObject *, PyObject *,
           PyObject *, PyObject *, PyObject *, PyObject *, int, PyObject *) = NULL;
static PyObject * (*gdb_PyObject_CallObject) (PyObject *callable_object, PyObject *args) = NULL;
static PyObject * (*gdb_PyObject_Call)(PyObject *callable_object, PyObject *args, PyObject *kw) = NULL;
static PyObject* (*gdb_PyUnicode_Decode)(const char *s, Py_ssize_t size, const char *encoding, const char *errors) = NULL;
static PyObject* (*gdb_PyUnicode_AsEncodedString)(register PyObject *unicode, const char *encoding, const char *errors) = NULL;
static PyObject* (*gdb_PyUnicode_FromEncodedObject)(register PyObject *obj, const char *encoding, const char *errors) = NULL;
static int *gdb_Py_DontWriteBytecodeFlag = NULL;



bool
is_python_available (void) {

  int i;
  static const char *libpython_names[] = {
#if HAVE_LIBPYTHON2_4
                     "libpython2.4.so.1.0", "libpython2.4.so.1",
#elif !defined(__APPLE__)
                     "libpython2.7.so.1.0", "libpython2.7.so.1",
                     "libpython2.6.so.1.0", "libpython2.6.so.1",
                     "libpython2.5.so.1.0", "libpython2.5.so.1",
#else
                     "libpython2.7.dylib",
                     "Python.framework/Versions/2.7/Python",
                     "/System/Library/Frameworks/Python.framework/Versions/2.7/Python",
#endif
                     NULL };

  if (python_initialized)
    return libpython_handle != NULL;

  for ( i = 0; libpython_names[i] && !libpython_handle ; ++i)
    libpython_handle = dlopen ( libpython_names[i], RTLD_NOW | RTLD_GLOBAL);

  python_initialized = true;
  if (!libpython_handle)
    return false;

#define RESOLVE_AND_CHECK(varname,type,symname)			\
  varname = (type) dlsym (libpython_handle, symname);	\
  if (!varname)							\
    {								\
      fprintf (stderr, "Symbol %s could not be found"		\
	       " in python library!\n", symname);		\
      goto err_out;						\
    }

  /* Resolve types and exceptions */
  RESOLVE_AND_CHECK(gdbpy_None, PyObject *, "_Py_NoneStruct");
  RESOLVE_AND_CHECK(gdbpy_True, PyObject *, "_Py_TrueStruct");
  RESOLVE_AND_CHECK(gdbpy_Zero, PyObject *, "_Py_ZeroStruct");

  RESOLVE_AND_CHECK(gdbpy_FloatType, PyTypeObject *, "PyFloat_Type");
  RESOLVE_AND_CHECK(gdbpy_BoolType, PyTypeObject *, "PyBool_Type");
  RESOLVE_AND_CHECK(gdbpy_IntType, PyTypeObject *, "PyInt_Type");
  RESOLVE_AND_CHECK(gdbpy_LongType, PyTypeObject *, "PyLong_Type");
  RESOLVE_AND_CHECK(gdbpy_SliceType, PyTypeObject *, "PySlice_Type");
  RESOLVE_AND_CHECK(gdbpy_StringType, PyTypeObject *, "PyString_Type");
  RESOLVE_AND_CHECK(gdbpy_ListType, PyTypeObject *, "PyList_Type");
  RESOLVE_AND_CHECK(gdbpy_TupleType, PyTypeObject *, "PyTuple_Type");
  RESOLVE_AND_CHECK(gdbpy_UnicodeType, PyTypeObject *, "PyUnicode_Type");
  RESOLVE_AND_CHECK(gdbpy_NotImplemented, PyObject *, "_Py_NotImplementedStruct");
  RESOLVE_AND_CHECK(gdbpy_ThreadState_Current, PyThreadState **, "_PyThreadState_Current");

  RESOLVE_AND_CHECK(pgdbpyExc_AttributeError, PyObject **, "PyExc_AttributeError");
  RESOLVE_AND_CHECK(pgdbpyExc_IndexError, PyObject **, "PyExc_IndexError");
  RESOLVE_AND_CHECK(pgdbpyExc_IOError, PyObject **, "PyExc_IOError");
  RESOLVE_AND_CHECK(pgdbpyExc_KeyError, PyObject **, "PyExc_KeyError");
  RESOLVE_AND_CHECK(pgdbpyExc_KeyboardInterrupt, PyObject **, "PyExc_KeyboardInterrupt");
  RESOLVE_AND_CHECK(pgdbpyExc_MemoryError, PyObject **, "PyExc_MemoryError");
  RESOLVE_AND_CHECK(pgdbpyExc_NotImplementedError, PyObject **, "PyExc_NotImplementedError");
  RESOLVE_AND_CHECK(pgdbpyExc_OverflowError, PyObject **, "PyExc_OverflowError");
  RESOLVE_AND_CHECK(pgdbpyExc_RuntimeError, PyObject **, "PyExc_RuntimeError");
  RESOLVE_AND_CHECK(pgdbpyExc_StopIteration, PyObject **, "PyExc_StopIteration");
  RESOLVE_AND_CHECK(pgdbpyExc_SystemError, PyObject **, "PyExc_SystemError");
  RESOLVE_AND_CHECK(pgdbpyExc_TypeError, PyObject **, "PyExc_TypeError");
  RESOLVE_AND_CHECK(pgdbpyExc_ValueError, PyObject **, "PyExc_ValueError");
  RESOLVE_AND_CHECK(pgdbpy_OSReadlineTState, PyThreadState **, "_PyOS_ReadlineTState");

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 4
     RESOLVE_AND_CHECK(pgdbpyOS_ReadlineFunctionPointer, char * (**) (FILE *, FILE *, const char *), "PyOS_ReadlineFunctionPointer");
#else
     RESOLVE_AND_CHECK(pgdbpyOS_ReadlineFunctionPointer, char * (**) (FILE *, FILE *, char *), "PyOS_ReadlineFunctionPointer");
#endif

  RESOLVE_AND_CHECK(pgdbpyExc_NameError, PyObject **, "PyExc_NameError");

  /* Resolve variadic functions */
  RESOLVE_AND_CHECK(gdbpy_Arg_UnpackTuple, int (*)(PyObject *, const char *, Py_ssize_t, Py_ssize_t, ...), "PyArg_UnpackTuple");
  RESOLVE_AND_CHECK(gdbpy_ErrFormat, PyObject * (*) (PyObject *, const char *, ...), "PyErr_Format");
  RESOLVE_AND_CHECK(gdbpy_BuildValue, PyObject * (*) (const char *, ...), STRINGIFY(Py_BuildValue));
  RESOLVE_AND_CHECK(gdbpy_PyObject_CallFunctionObjArgs, PyObject * (*) (PyObject *,...), "PyObject_CallFunctionObjArgs");
  RESOLVE_AND_CHECK(gdbpy_PyObject_CallMethodObjArgs, PyObject * (*) (PyObject *, PyObject *,...), "PyObject_CallMethodObjArgs");
  RESOLVE_AND_CHECK(gdbpy_PyObject_CallMethod, PyObject * (*) (PyObject *o, const char *m, const char *format, ...), "PyObject_CallMethod");
  RESOLVE_AND_CHECK(gdbpy_PyArg_ParseTuple, int (*) (PyObject *obj, const char *, ...), STRINGIFY(PyArg_ParseTuple));
  RESOLVE_AND_CHECK(gdbpy_PyArg_ParseTupleAndKeywords, int (*) (PyObject *obj, PyObject *, const char *, char **, ...), STRINGIFY(PyArg_ParseTupleAndKeywords));
  RESOLVE_AND_CHECK(gdbpy_PyArg_VaParseTupleAndKeywords, int (*) (PyObject *obj, PyObject *, const char *, char **, ...), STRINGIFY(PyArg_VaParseTupleAndKeywords));
  RESOLVE_AND_CHECK(gdbpy_StringFromFormat, PyObject * (*)  (const char *, ...), "PyString_FromFormat");

#define RESOLVE(varname,type,symname)				\
  varname = (type) dlsym (libpython_handle, symname);

  /* Resolve functions */
  RESOLVE(gdbpy_Dict_SetItem, int (*) (PyObject *mp, PyObject *key, PyObject *item), "PyDict_SetItem");
  RESOLVE(gdbpy_Dict_Keys, PyObject * (*) (PyObject *mp), "PyDict_Keys");
  RESOLVE(gdbpy_Err_SetNone, void (*) (PyObject *), "PyErr_SetNone");
  RESOLVE(gdbpy_PySys_GetObject, PyObject * (*) (const char *), "PySys_GetObject");
  RESOLVE(gdbpy_PySys_SetPath, void (*) (GDB_PYSYS_SETPATH_CHAR *), "PySys_SetPath");
  RESOLVE(gdbpy_PyInt_AsSize_t, Py_ssize_t (*) (PyObject *), "PyInt_AsSize_t");
  RESOLVE(gdbpy_PyInt_FromSize_t, PyObject * (*) (size_t), "PyInt_FromSize_t");
  RESOLVE(gdbpy_PyInt_FromLong, PyObject * (*) (long), "PyInt_FromLong");
  RESOLVE(gdbpy_PyObject_GetBuffer, int (*) (PyObject *, Py_buffer *, int), "PyObject_GetBuffer");
  RESOLVE(gdbpy_PyBuffer_Release, void (*) (Py_buffer *buf), "PyBuffer_Release");

  RESOLVE(gdbpy_Long_AsLong, PY_LONG_LONG (*) (PyObject *), "PyLong_AsLong");
  RESOLVE(gdbpy_PyObject__IsInstance, int (*) (PyObject *object, PyObject *typeorclass), "PyObject_IsInstance");
  RESOLVE(gdbpy_PySequence_Concat, PyObject * (*) (PyObject *o1, PyObject *o2), "PySequence_Concat");
  RESOLVE(gdbpy_PyErr_NewException, PyObject * (*) (const char *name, PyObject *base, PyObject *dict), "PyErr_NewException");
  RESOLVE(gdbpy_PyString_FromStringAndSize, PyObject * (*) (const char *, Py_ssize_t), "PyString_FromStringAndSize");
  RESOLVE(gdbpy_PyBytes_FromStringAndSize, PyObject * (*) (const char *, Py_ssize_t), "PyBytes_FromStringAndSize");
#ifdef IS_PY3K
  RESOLVE(gdbpy_PyMemoryView_FromObject, PyObject * (*) (PyObject *), "PyMemoryView_FromObject");
  RESOLVE(gdbpy_PySlice_GetIndicesEx, int (*) (PyObject *slice, Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step, Py_ssize_t *slicelength), "PySlice_GetIndiciesEx");
#else
  RESOLVE(gdbpy_PyBuffer_FromObject, PyObject * (*) (PyObject *, Py_ssize_t offset, Py_ssize_t size), "PyBuffer_FromObject");
  RESOLVE(gdbpy_PySlice_GetIndicesEx, int (*) (PySliceObject *slice, Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step, Py_ssize_t *slicelength), "PySlice_GetIndiciesEx");
#endif
  RESOLVE(gdbpy_PySlice_Check, int (*) (PyObject *slice), "PySlice_Check");

#ifdef HAVE_LONG_LONG

  RESOLVE(gdbpy_PyLong_FromLongLong, PyObject * (*) (long long val), "PyLong_FromLongLong");
  RESOLVE(gdbpy_PyLong_FromUnsignedLongLong, PyObject * (*) (unsigned long long val), "PyLong_FromUnsignedLongLong");
  RESOLVE(gdbpy_PyLong_AsUnsignedLongLong, unsigned long long (*) (PyObject *obj), "PyLong_AsUnsignedLongLong");

extern unsigned long long (*gdbpy_PyLong_AsUnsignedLongLong) (PyObject *obj);

#else /* HAVE_LONG_LONG */

  RESOLVE(gdbpy_PyLong_FromLongLong, PyObject * (*) (long  val), "PyLong_FromLongLong");
  RESOLVE(gdbpy_PyLong_FromUnsignedLongLong, PyObject * (*) (unsigned long val), "PyLong_FromUnsignedLongLong");
  RESOLVE(gdbpy_PyLong_AsUnsignedLongLong, unsigned long (*) (PyObject *obj), "PyLong_AsUnsignedLongLong");

#endif /* HAVE_LONG_LONG */
  
  /* Resolve indirectly called functions */

  RESOLVE(gdb_PyBool_FromLong, PyObject * (*) (long), "PyBool_FromLong");
  RESOLVE(gdb_PyBuffer_FromReadWriteObject, PyObject * (*) (PyObject *base, Py_ssize_t offset, Py_ssize_t size), "PyBuffer_FromReadWriteObject");
  RESOLVE(gdb_PyCallable_Check, int (*) (PyObject *o), "PyCallable_Check");
  RESOLVE(gdb_PyDict_New, PyObject * (*) (void), "PyDict_New");
  RESOLVE(gdb_PyDict_SetItemString, int (*) (PyObject *dp, const char *key, PyObject *item), "PyDict_SetItemString");
  RESOLVE(gdb_PyErr_Clear, void (*) (void), "PyErr_Clear");
  RESOLVE(gdb_PyErr_ExceptionMatches, int (*) (PyObject *), "PyErr_ExceptionMatches");
  RESOLVE(gdb_PyErr_Fetch, void (*) (PyObject **, PyObject **, PyObject **), "PyErr_Fetch");
  RESOLVE(gdb_PyErr_GivenExceptionMatches, int (*) (PyObject *, PyObject *), "PyErr_GivenExceptionMatch");
  RESOLVE(gdb_PyErr_Occurred, PyObject * (*) (void), "PyErr_Occurred");
  RESOLVE(gdb_PyErr_Print, void (*) (void), "PyErr_Print");
  RESOLVE(gdb_PyErr_Restore, void (*) (PyObject *, PyObject *, PyObject *), "PyErr_Restore");
  RESOLVE(gdb_PyErr_SetFromErrno, PyObject * (*) (PyObject *), "PyErr_SetFromErrno");
  RESOLVE(gdb_PyErr_SetInterrupt, void (*) (void), "PyErr_SetInterrupt");
  RESOLVE(gdb_PyErr_SetObject, void (*) (PyObject *, PyObject *), "PyErr_SetObject");
  RESOLVE(gdb_PyErr_SetString, void (*) (PyObject *, const char *), "PyErr_SetString");
  //RESOLVE(gdb_PyErr_NewException, PyObject * (*) (const char *name, PyObject *base, PyObject *dict), "PyErr_NewException");
  RESOLVE(gdb_PyEval_InitThreads, void (*) (void), "PyEval_InitThreads");
  RESOLVE(gdb_PyEval_ReleaseLock, void (*) (void), "PyEval_ReleaseLock");
  RESOLVE(gdb_PyEval_RestoreThread, void (*) (PyThreadState *), "PyEval_RestoreThread");
  RESOLVE(gdb_PyEval_SaveThread, PyThreadState * (*) (void), "PyEval_SaveThread");
  RESOLVE(gdb_PyFloat_AsDouble, double (*) (PyObject *), "PyFloat_AsDouble");
  RESOLVE(gdb_PyFloat_FromDouble, PyObject * (*) (double), "PyFloat_FromDouble");
  RESOLVE(gdb_PyGILState_Ensure, PyGILState_STATE (*) (void), "PyGILState_Ensure");
  RESOLVE(gdb_PyGILState_Release, void (*) (PyGILState_STATE), "PyGILState_Release");
  RESOLVE(gdb_PyImport_AddModule, PyObject * (*) (const char *), "PyImport_AddModule");
  RESOLVE(gdb_PyImport_ImportModule, PyObject * (*) (const char *), "PyImport_ImportModule");
  RESOLVE(gdb_PyInt_AsLong, long (*) (PyObject *), "PyInt_AsLong");
  RESOLVE(gdb_PyInt_FromLong, PyObject * (*) (long), "PyInt_FromLong");
  RESOLVE(gdb_PyInt_GetMax, long (*) (void), "PyInt_GetMax");
  RESOLVE(gdb_PyIter_Next, PyObject * (*) (PyObject *), "PyIter_Next");
  RESOLVE(gdb_PyList_Append, int (*) (PyObject *, PyObject *), "PyList_Append");
  RESOLVE(gdb_PyList_AsTuple, PyObject * (*) (PyObject *), "PyList_AsTuple");
  RESOLVE(gdb_PyList_GetItem, PyObject * (*) (PyObject *, Py_ssize_t), "PyList_GetItem");
  RESOLVE(gdb_PyList_Insert, int (*) (PyObject *, Py_ssize_t, PyObject *), "PyList_Insert");
  RESOLVE(gdb_PyList_New, PyObject * (*) (Py_ssize_t size), "PyList_New");
  RESOLVE(gdb_PyList_Size, Py_ssize_t (*) (PyObject *), "PyList_Size");
  RESOLVE(gdb_PyLong_AsLongLong, PY_LONG_LONG (*) (PyObject *), "PyLong_AsLongLong");
  RESOLVE(gdb_PyLong_AsUnsignedLongLong, unsigned PY_LONG_LONG (*) (PyObject *), "PyLong_AsUnsignedLongLong");
  RESOLVE(gdb_PyLong_FromLong, PyObject * (*) (long), "PyLong_FromLong");
  RESOLVE(gdb_PyLong_FromLongLong, PyObject * (*) (PY_LONG_LONG), "PyLong_FromLongLong");
  RESOLVE(gdb_PyLong_FromUnsignedLong, PyObject * (*) (unsigned long), "PyLong_FromUnsignedLong");
  RESOLVE(gdb_PyLong_FromUnsignedLongLong, PyObject * (*) (unsigned PY_LONG_LONG), "PyLong_FromUnsignedLongLong");
  RESOLVE(gdb_PyMem_Malloc, void * (*) (size_t), "PyMem_Malloc");
  RESOLVE(gdb_PyModule_AddIntConstant, int (*) (PyObject *, const char *, long), "PyModule_AddIntConstant");
  RESOLVE(gdb_PyModule_AddObject, int (*) (PyObject *, const char *, PyObject *), "PyModule_AddObject");
  RESOLVE(gdb_PyModule_AddStringConstant, int (*) (PyObject *, const char *, const char *), "PyModule_AddStringConstant");
  RESOLVE(gdb_PyModule_GetDict, PyObject * (*) (PyObject *), "PyModule_GetDict");
  RESOLVE(gdb_PyNumber_Long, PyObject * (*) (PyObject *), "PyNumber_Long");
  RESOLVE(gdb_PyOS_InterruptOccurred, int (*) (void), "PyOS_InterruptOccurred");
  RESOLVE(gdb_PyObject_AsReadBuffer, int (*) (PyObject *obj, const void **, Py_ssize_t *), "PyObject_AsReadBuffer");
  RESOLVE(gdb_PyObject_CheckReadBuffer, int (*) (PyObject *), "PyObject_CheckReadBuffer");
  RESOLVE(gdb_PyObject_GenericGetAttr, PyObject * (*) (PyObject *, PyObject *), "PyObject_GenericGetAttr");
  RESOLVE(gdb_PyObject_GenericSetAttr, int (*) (PyObject *arg1, PyObject *arg2, PyObject *arg3), "PyObject_GenericSetAttr");
  RESOLVE(gdb_PyObject_GetAttr, PyObject * (*) (PyObject *, PyObject *), "PyObject_GetAttr");
  RESOLVE(gdb_PyObject_GetAttrString, PyObject * (*) (PyObject *, const char *), "PyObject_GetAttrString");
  RESOLVE(gdb_PyObject_GetIter, PyObject * (*) (PyObject *), "PyObject_GetIter");
  RESOLVE(gdb_PyObject_HasAttr, int (*) (PyObject *, PyObject *), "PyObject_HasAttr");
  RESOLVE(gdb_PyObject_HasAttrString,  int (*) (PyObject *, const char *), "PyObject_HasAttrString");
  RESOLVE(gdb_PyObject_IsTrue, int (*) (PyObject *), "PyObject_IsTrue");
  RESOLVE(gdb_PyObject_RichCompareBool, int (*) (PyObject *, PyObject *, int), "PyObject_RichCompareBool");
  RESOLVE(gdb_PyObject_SetAttrString, int (*) (PyObject *, const char *, PyObject *), "PyObject_SetAttrString");
  RESOLVE(gdb_PyObject_Str, PyObject * (*) (PyObject *), "PyObject_Str");
  RESOLVE(gdb_PyRun_InteractiveLoopFlags, int (*) (FILE *, const char *, PyCompilerFlags *), "PyRun_InteractiveLoopFlags");
  RESOLVE(gdb_PyRun_StringFlags, PyObject * (*) (const char *, int, PyObject *, PyObject *, PyCompilerFlags *), "PyRun_StringFlags");
  RESOLVE(gdb_PyRun_SimpleFileExFlags, int (*) (FILE *, const char *, int, PyCompilerFlags *), "PyRun_SimpleFileExFlags");
  RESOLVE(gdb_PyRun_SimpleStringFlags, int (*) (const char *, PyCompilerFlags *), "PyRun_SimpleStringFlags");
  RESOLVE(gdb_PySequence_Check, int (*) (PyObject *), "PySequence_Check");
  RESOLVE(gdb_PySequence_DelItem, int (*) (PyObject *o, Py_ssize_t i), "PySequence_DelItem");
  RESOLVE(gdb_PySequence_GetItem, PyObject * (*) (PyObject *o, Py_ssize_t i), "PySequence_GetItem");
  RESOLVE(gdb_PySequence_Index, Py_ssize_t (*) (PyObject *o, PyObject *value), "PySequence_Index");
  RESOLVE(gdb_PySequence_List, PyObject * (*) (PyObject *o), "PySequence_List");
  RESOLVE(gdb_PySequence_Size, Py_ssize_t (*) (PyObject *o), "PySequence_Size");
  RESOLVE(gdb_PyString_AsString, char * (*) (PyObject *o), "PyString_AsString");
  RESOLVE(gdb_PyString_Decode, PyObject * (*) (const char *, Py_ssize_t, const char *, const char *), "PyString_Decode");
  RESOLVE(gdb_PyString_FromString, PyObject * (*) (const char *), "PyString_FromString");
  RESOLVE(gdb_PyString_Size, Py_ssize_t (*) (PyObject *), "PyString_Size");
  //RESOLVE(gdb_PySys_GetObject, PyObject * (*) (char *), "PySys_GetObject");
  //RESOLVE(gdb_PySys_SetPath, void (*) (char *), "PySys_SetPath");
  RESOLVE(gdb_PyThreadState_Get, PyThreadState * (*) (void), "PyThreadState_Get");
  RESOLVE(gdb_PyThreadState_Swap, PyThreadState * (*) (PyThreadState *), "PyThreadState_Swap");
  RESOLVE(gdb_PyTuple_GetItem, PyObject * (*) (PyObject *, Py_ssize_t), "PyTuple_GetItem");
  RESOLVE(gdb_PyTuple_New, PyObject * (*) (Py_ssize_t), "PyTuple_New");
  RESOLVE(gdb_PyTuple_SetItem, int (*)  (PyObject *, Py_ssize_t, PyObject *), "PyTuple_SetItem");
  RESOLVE(gdb_PyTuple_Size, Py_ssize_t (*) (PyObject *), "PyTuple_Size");
  RESOLVE(gdb_PyType_GenericNew, PyObject * (*) (PyTypeObject *, PyObject *, PyObject *), "PyType_GenericNew");
  RESOLVE(gdb_PyType_IsSubtype, int (*) (PyTypeObject *, PyTypeObject *), "PyType_IsSubtype");
  RESOLVE(gdb_PyType_Ready, int (*) (PyTypeObject *), "PyType_Ready");
  RESOLVE(gdb_Py_Finalize, void (*) (void), "Py_Finalize");
  RESOLVE(gdb_Py_FlushLine, int (*) (void), "Py_FlushLine");
  RESOLVE(gdb_Py_Initialize, void (*) (void), "Py_Initialize");
  RESOLVE(gdb_Py_InitModule4, PyObject * (*) (const char *, PyMethodDef *, const char *, PyObject *, int), "Py_InitModule4");
  RESOLVE(gdb_Py_InitModule4_64, PyObject * (*) (const char *, PyMethodDef *, const char *, PyObject *, int), "Py_InitModule4_64");
  RESOLVE(gdb_PyObject_Call, PyObject * (*) (PyObject *callable_object, PyObject *args, PyObject *kw), "PyObject_Call");
  RESOLVE(gdb_PyObject_CallObject, PyObject * (*) (PyObject *callable_object, PyObject *args), "PyObject_CallObject");
  RESOLVE(gdb_Py_SetProgramName, void (*) (char *), "Py_SetProgramName");
  RESOLVE(gdb__PyObject_New, PyObject * (*) (PyTypeObject *), "_PyObject_New");
  RESOLVE(gdb_PyCode_New, PyCodeObject * (*) (int, int, int, int,
					      PyObject *, PyObject *, PyObject *, PyObject *,
					      PyObject *, PyObject *, PyObject *, PyObject *, int, PyObject *), "PyCode_New");
  RESOLVE(gdb_PyFrame_New, PyFrameObject * (*) (PyThreadState *, PyCodeObject *, PyObject *, PyObject *), "PyFrame_New");
#ifdef __APPLE__
  RESOLVE(gdb_PyUnicode_Decode, PyObject * (*) (const char *s, Py_ssize_t size, const char *encoding, const char *errors), "PyUnicodeUCS2_Decode");
  RESOLVE(gdb_PyUnicode_AsEncodedString, PyObject * (*) (register PyObject *unicode, const char *encoding, const char *errors), "PyUnicodeUCS2_AsEncodedString");
  RESOLVE(gdb_PyUnicode_FromEncodedObject, PyObject * (*) (register PyObject *obj, const char *encoding, const char *errors), "PyUnicodeUCS2_FromEncodedObject");
#else
  RESOLVE(gdb_PyUnicode_Decode, PyObject * (*) (const char *s, Py_ssize_t size, const char *encoding, const char *errors), "PyUnicodeUCS4_Decode");
  RESOLVE(gdb_PyUnicode_AsEncodedString, PyObject * (*) (register PyObject *unicode, const char *encoding, const char *errors), "PyUnicodeUCS4_AsEncodedString");
  RESOLVE(gdb_PyUnicode_FromEncodedObject, PyObject * (*) (register PyObject *obj, const char *encoding, const char *errors), "PyUnicodeUCS4_FromEncodedObject");
#endif
  RESOLVE(gdb_Py_DontWriteBytecodeFlag, int *, "Py_DontWriteBytecodeFlag");
  return true;
err_out:
  dlclose (libpython_handle);
  libpython_handle = NULL;
  return false;
}

#define PYWRAPPER(rtype,name)                                                 \
rtype                                                                         \
name (void)                                                                   \
{                                                                             \
  if (!is_python_available () || gdb_## name == NULL)                         \
    {                                                                         \
      warning ("%s: called while Python is not available!", __FUNCTION__);    \
      return (rtype) 0;                                                       \
    }                                                                         \
  return gdb_## name ();                                                      \
}

#define PYWRAPPER_ARG1(rtype,name, atype)                                     \
rtype                                                                         \
name (atype arg1)                                                             \
{                                                                             \
  if (!is_python_available () || gdb_## name == NULL)                         \
    {                                                                         \
      warning ("%s: called while Python is not available!", __FUNCTION__);    \
      return (rtype) 0;                                                       \
    }                                                                         \
  return gdb_## name (arg1);                                                  \
}

#define PYWRAPPER_ARG2(rtype,name, atype, btype)                              \
rtype                                                                         \
name (atype arg1, btype arg2)                                                 \
{                                                                             \
  if (!is_python_available () || gdb_## name == NULL)                         \
    {                                                                         \
      warning ("%s: called while Python is not available!", __FUNCTION__);    \
      return (rtype) 0;                                                       \
    }                                                                         \
  return gdb_## name (arg1, arg2);                                            \
}

#define PYWRAPPER_ARG3(rtype,name, atype, btype, ctype)                       \
rtype                                                                         \
name (atype arg1, btype arg2, ctype arg3)                                     \
{                                                                             \
  if (!is_python_available () || gdb_## name == NULL)                         \
    {                                                                         \
      warning ("%s: called while Python is not available!", __FUNCTION__);    \
      return (rtype) 0;                                                       \
    }                                                                         \
  return gdb_## name (arg1, arg2, arg3);                                      \
}

#define PYWRAPPER_ARG4(rtype,name, atype, btype, ctype, dtype)                \
rtype                                                                         \
name (atype arg1, btype arg2, ctype arg3, dtype arg4)                         \
{                                                                             \
  if (!is_python_available () || gdb_## name == NULL)                         \
    {                                                                         \
      warning ("%s: called while Python is not available!", __FUNCTION__);    \
      return (rtype) 0;                                                       \
    }                                                                         \
  return gdb_## name (arg1, arg2, arg3, arg4);                                \
}

#define PYWRAPPERVOID(name)                                                   \
void                                                                          \
name (void)                                                                   \
{                                                                             \
  if (is_python_available () && gdb_## name != NULL)                          \
    gdb_## name ();                                                           \
  else                                                                        \
    warning ("%s: called while Python is not available!", __FUNCTION__);      \
}

#define PYWRAPPERVOID_ARG1(name, atype)                                       \
void                                                                          \
name (atype arg1)                                                             \
{                                                                             \
  if (is_python_available () && gdb_## name != NULL)                          \
    gdb_## name (arg1);                                                       \
  else                                                                        \
    warning ("%s: called while Python is not available!", __FUNCTION__);      \
}

#define PYWRAPPERVOID_ARG2(name, atype, btype)                                \
void                                                                          \
name (atype arg1, btype arg2)                                                 \
{                                                                             \
  if (is_python_available () && gdb_## name != NULL)                          \
    gdb_## name (arg1, arg2);                                                 \
  else                                                                        \
    warning ("%s: called while Python is not available!", __FUNCTION__);      \
}

#define PYWRAPPERVOID_ARG3(name, atype, btype, ctype)                         \
void                                                                          \
name (atype arg1, btype arg2, ctype arg3)                                     \
{                                                                             \
  if (is_python_available () && gdb_## name != NULL)                          \
    gdb_## name (arg1, arg2, arg3);                                           \
  else                                                                        \
    warning ("%s: called while Python is not available!", __FUNCTION__);      \
}

PYWRAPPER_ARG1(PyObject *, PyBool_FromLong, long)
PYWRAPPER_ARG1(int, PyCallable_Check, PyObject *)
PYWRAPPER (PyObject *, PyDict_New)
PYWRAPPER_ARG3(int, PyDict_SetItemString, PyObject *, const char *, PyObject *)
PYWRAPPER_ARG1(int, PyErr_ExceptionMatches, PyObject *)
PYWRAPPER_ARG2(int, PyErr_GivenExceptionMatches, PyObject *, PyObject *)
PYWRAPPER (PyObject *, PyErr_Occurred)
PYWRAPPER_ARG1(PyObject *, PyErr_SetFromErrno, PyObject *)
PYWRAPPERVOID(PyErr_Clear)
PYWRAPPERVOID_ARG3(PyErr_Fetch, PyObject **, PyObject **, PyObject **)
PYWRAPPERVOID(PyErr_Print)
PYWRAPPERVOID_ARG3(PyErr_Restore, PyObject *, PyObject *, PyObject *)
PYWRAPPERVOID(PyErr_SetInterrupt)
PYWRAPPERVOID_ARG2(PyErr_SetObject, PyObject *, PyObject *)
PYWRAPPERVOID_ARG2(PyErr_SetString, PyObject *, const char *)
PYWRAPPERVOID(PyEval_InitThreads)
PYWRAPPERVOID(PyEval_ReleaseLock)
PYWRAPPERVOID_ARG1(PyEval_RestoreThread, PyThreadState *)
PYWRAPPER (PyThreadState *, PyEval_SaveThread);
PYWRAPPER_ARG1(double, PyFloat_AsDouble, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyFloat_FromDouble, double)
PYWRAPPER (PyGILState_STATE, PyGILState_Ensure)
PYWRAPPERVOID_ARG1(PyGILState_Release, PyGILState_STATE)
PYWRAPPER_ARG1(PyObject *, PyImport_AddModule, CONSTCHAR *)
PYWRAPPER_ARG1(PyObject *,PyImport_ImportModule, CONSTCHAR *)
PYWRAPPER_ARG1(long, PyInt_AsLong, PyObject *)
PYWRAPPER_ARG1(PyObject *,PyInt_FromLong, long)
PYWRAPPER (long, PyInt_GetMax)
PYWRAPPER_ARG1(PyObject *, PyIter_Next, PyObject *)
PYWRAPPER_ARG2(int, PyList_Append, PyObject *, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyList_AsTuple, PyObject *)
PYWRAPPER_ARG2(PyObject *,PyList_GetItem, PyObject *, Py_ssize_t)
PYWRAPPER_ARG3(int, PyList_Insert, PyObject *, Py_ssize_t, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyList_New, Py_ssize_t)
PYWRAPPER_ARG1(Py_ssize_t, PyList_Size, PyObject *)
PYWRAPPER_ARG1(PY_LONG_LONG, PyLong_AsLongLong, PyObject *)
PYWRAPPER_ARG1(unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyLong_FromLong, long)
PYWRAPPER_ARG1(PyObject *, PyLong_FromLongLong, PY_LONG_LONG)
PYWRAPPER_ARG1(PyObject *, PyLong_FromUnsignedLong, unsigned long)
PYWRAPPER_ARG1(PyObject *, PyLong_FromUnsignedLongLong, unsigned PY_LONG_LONG)
PYWRAPPER_ARG1(void *, PyMem_Malloc, size_t)
PYWRAPPER_ARG3(int, PyModule_AddIntConstant, PyObject *, CONSTCHAR *, long)
PYWRAPPER_ARG3(int, PyModule_AddObject, PyObject *, CONSTCHAR *, PyObject *)
PYWRAPPER_ARG3(int, PyModule_AddStringConstant, PyObject *, CONSTCHAR *, CONSTCHAR *)
PYWRAPPER_ARG1(PyObject *, PyModule_GetDict, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyNumber_Long, PyObject *)
PYWRAPPER (int, PyOS_InterruptOccurred)
PYWRAPPER_ARG1(int, PyObject_CheckReadBuffer, PyObject *)
PYWRAPPER_ARG2(PyObject *, PyObject_GenericGetAttr, PyObject *, PyObject *)
PYWRAPPER_ARG2(PyObject *, PyObject_GetAttr, PyObject *, PyObject *)
PYWRAPPER_ARG2(PyObject *, PyObject_GetAttrString, PyObject *, CONSTCHAR *)
PYWRAPPER_ARG1(PyObject *, PyObject_GetIter, PyObject *)
PYWRAPPER_ARG2(int, PyObject_HasAttr, PyObject *, PyObject *)
PYWRAPPER_ARG2(int, PyObject_HasAttrString, PyObject *, CONSTCHAR *)
PYWRAPPER_ARG1(int, PyObject_IsTrue, PyObject *)
PYWRAPPER_ARG3(int, PyObject_RichCompareBool, PyObject *, PyObject *, int)
PYWRAPPER_ARG3(int, PyObject_SetAttrString, PyObject *, CONSTCHAR *, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyObject_Str, PyObject *)
PYWRAPPER_ARG3(int, PyRun_InteractiveLoopFlags, FILE *, const char *, PyCompilerFlags *)
PYWRAPPER_ARG4(int, PyRun_SimpleFileExFlags, FILE *, const char *, int, PyCompilerFlags *)
PYWRAPPER_ARG2(int, PyRun_SimpleStringFlags, const char *, PyCompilerFlags *)
PYWRAPPER_ARG1(int, PySequence_Check, PyObject *);
PYWRAPPER_ARG2(int, PySequence_DelItem, PyObject *, Py_ssize_t)
PYWRAPPER_ARG2(PyObject *, PySequence_GetItem, PyObject *, Py_ssize_t)
PYWRAPPER_ARG2(Py_ssize_t, PySequence_Index, PyObject *, PyObject *)
PYWRAPPER_ARG1(PyObject *, PySequence_List, PyObject *)
PYWRAPPER_ARG1(Py_ssize_t, PySequence_Size, PyObject *)
PYWRAPPER_ARG1(char *, PyString_AsString, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyString_FromString, const char *)
PYWRAPPER_ARG1(Py_ssize_t, PyString_Size, PyObject *)
//PYWRAPPER_ARG1(PyObject *, PySys_GetObject, char *)
//PYWRAPPERVOID_ARG1(PySys_SetPath,char *)
PYWRAPPER (PyThreadState *, PyThreadState_Get);
PYWRAPPER_ARG1(PyThreadState *, PyThreadState_Swap, PyThreadState *)
PYWRAPPER_ARG2(PyObject *, PyTuple_GetItem, PyObject *, Py_ssize_t)
PYWRAPPER_ARG1(PyObject *, PyTuple_New, Py_ssize_t)
PYWRAPPER_ARG3(int, PyTuple_SetItem, PyObject *, Py_ssize_t, PyObject *)
PYWRAPPER_ARG1(Py_ssize_t, PyTuple_Size, PyObject *)
PYWRAPPER_ARG2(int, PyType_IsSubtype, PyTypeObject *, PyTypeObject *)
PYWRAPPER_ARG1(int, PyType_Ready, PyTypeObject *)
PYWRAPPERVOID(Py_Finalize)
PYWRAPPER(int, Py_FlushLine)
PYWRAPPERVOID_ARG1(Py_SetProgramName, char *)
PYWRAPPER_ARG1(PyObject *, _PyObject_New, PyTypeObject *)
PYWRAPPER_ARG2(PyObject *, PyObject_CallObject, PyObject *, PyObject *)
PYWRAPPER_ARG3(PyObject *, PyObject_Call, PyObject *, PyObject *, PyObject *)
//PYWRAPPER_ARG3(PyObject *, PyErr_NewException, char *, PyObject *, PyObject *)
PYWRAPPER_ARG4(PyObject *, PyString_Decode, const char *, Py_ssize_t, const char *, const char *)
PYWRAPPER_ARG3(PyObject *, PyType_GenericNew, PyTypeObject *, PyObject *, PyObject *)
PYWRAPPER_ARG3(int, PyObject_AsReadBuffer, PyObject *, const void **, Py_ssize_t *)
PYWRAPPER_ARG3(int, PyObject_GenericSetAttr, PyObject *, PyObject *, PyObject *)
PYWRAPPER_ARG3(PyObject *, PyBuffer_FromReadWriteObject, PyObject *, Py_ssize_t, Py_ssize_t)
PYWRAPPER_ARG4 (PyObject *, PyUnicode_Decode, const char *, Py_ssize_t, const char *, const char *)
PYWRAPPER_ARG3 (PyObject *, PyUnicode_FromEncodedObject, register PyObject *, const char *, const char *)
PYWRAPPER_ARG3 (PyObject *,PyUnicode_AsEncodedString, register PyObject *, const char *, const char *)
PYWRAPPER_ARG4(PyFrameObject *, PyFrame_New, PyThreadState *,PyCodeObject *, PyObject *, PyObject *)

void
Py_Initialize(void)
{
  if (!is_python_available () || gdb_Py_Initialize == NULL)
    {
        warning ("%s: called while Python is not available!", __FUNCTION__);
      return;
    }
  if (gdb_Py_DontWriteBytecodeFlag != NULL)
    *gdb_Py_DontWriteBytecodeFlag = 1;
  gdb_Py_Initialize();
}


PyCodeObject *
PyCode_New(int a, int b, int c, int d,
           PyObject *e, PyObject *f, PyObject *g, PyObject *h,
           PyObject *i, PyObject *j, PyObject *k, PyObject *l, int m, PyObject *n)
{
  if (!is_python_available () || gdb_PyCode_New == NULL)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return NULL;
    }
  return gdb_PyCode_New(a, b, c, d, e, f, g, h, i, j, k, l, m, n);
}

PyObject *
PyRun_StringFlags(const char *arg1, int arg2, PyObject *arg3, PyObject *arg4, PyCompilerFlags *arg5)
{
  if (!is_python_available () || gdb_PyRun_StringFlags == NULL)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return NULL;
    }
  return gdb_PyRun_StringFlags (arg1, arg2, arg3, arg4, arg5);
}

PyObject*
Py_InitModule4(CONSTCHAR *name, PyMethodDef *methods, CONSTCHAR *doc, PyObject *self, int apiver)
{
  if (!is_python_available () || (gdb_Py_InitModule4 == NULL && gdb_Py_InitModule4_64 == NULL))
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return NULL;
    }
  /* For 64-bit processes, entry point name have changed
   * in more recent version of libpython 
   */
  if (gdb_Py_InitModule4_64 != NULL)
    return gdb_Py_InitModule4_64 (name, methods, doc, self, apiver);
  return gdb_Py_InitModule4 (name, methods, doc, self, apiver);
}


#if HAVE_LIBPYTHON2_4
PyObject *
PyRun_String(const char *arg1, int arg2, PyObject *arg3, PyObject *arg4)
{
  if (!is_python_available () || gdb_PyRun_StringFlags == 0)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return NULL;
    }
  return gdb_PyRun_StringFlags (arg1, arg2, arg3, arg4, NULL);
}

int
PyRun_SimpleString(const char * arg1) {
  if (!is_python_available () || gdb_PyRun_SimpleStringFlags == NULL)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return 0;
    }
  return gdb_PyRun_SimpleStringFlags(arg1, NULL);
}

int
PyRun_InteractiveLoop(FILE * arg1,const char * arg2) {
  if (!is_python_available () || gdb_PyRun_InteractiveLoopFlags == NULL)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return 0;
    }
  return gdb_PyRun_InteractiveLoopFlags(arg1, arg2, NULL);
}

int
PyRun_SimpleFile(FILE * arg1,const char * arg2) {
  if (!is_python_available () || gdb_PyRun_SimpleFileExFlags == NULL)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return 0;
    }
  return gdb_PyRun_SimpleFileExFlags(arg1, arg2, 0, NULL);
}
#endif
