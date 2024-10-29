#include <Python.h>
#include <iostream>
#include "pytensor_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
//////////////////////
////  Support Code
//////////////////////
#ifndef MIN_LONG
#define MIN_LONG NPY_MIN_LONG
#endif

    namespace {
    struct __struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49 {
        PyObject* __ERROR;

        PyObject* storage_V5;
PyObject* storage_V3;
PyObject* storage_V1;
        

        __struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49() {
            // This is only somewhat safe because we:
            //  1) Are not a virtual class
            //  2) Do not use any virtual classes in the members
            //  3) Deal with mostly POD and pointers

            // If this changes, we would have to revise this, but for
            // now I am tired of chasing segfaults because
            // initialization code had an error and some pointer has
            // a junk value.
            #ifndef PYTENSOR_DONT_MEMSET_STRUCT
            memset(this, 0, sizeof(*this));
            #endif
        }
        ~__struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V5, PyObject* storage_V3, PyObject* storage_V1) {
            Py_XINCREF(storage_V5);
Py_XINCREF(storage_V3);
Py_XINCREF(storage_V1);
            this->storage_V5 = storage_V5;
this->storage_V3 = storage_V3;
this->storage_V1 = storage_V1;
            




            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_3:

double __DUMMY_3;
__label_5:

double __DUMMY_5;
__label_8:

double __DUMMY_8;

            Py_XDECREF(this->storage_V5);
Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V1);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
    
        PyArrayObject* V1;
        
    PyObject* py_V3;
    
        PyArrayObject* V3;
        
    PyObject* py_V5;
    
        PyArrayObject* V5;
        
{

    py_V1 = PyList_GET_ITEM(storage_V1, 0);
    {Py_XINCREF(py_V1);}
    
        if (py_V1 == Py_None)
        {
            
        V1 = NULL;
        
        }
        else
        {
            
        V1 = (PyArrayObject*)(py_V1);
        Py_XINCREF(V1);
        
        }
        
{

    py_V3 = PyList_GET_ITEM(storage_V3, 0);
    {Py_XINCREF(py_V3);}
    
        V3 = (PyArrayObject*)(py_V3);
        Py_XINCREF(V3);
        
{

    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
        V5 = (PyArrayObject*)(py_V5);
        Py_XINCREF(V5);
        
{
// Op class AdvancedSubtensor1

            PyArrayObject *indices;
            int i_type = PyArray_TYPE(V3);
            if (i_type != NPY_INTP) {
                // Cast V3 to NPY_INTP (expected by PyArray_TakeFrom),
                // if all values fit.
                if (!PyArray_CanCastSafely(i_type, NPY_INTP) &&
                    PyArray_SIZE(V3) > 0) {
                    npy_int64 min_val, max_val;
                    PyObject* py_min_val = PyArray_Min(V3, NPY_MAXDIMS,
                                                       NULL);
                    if (py_min_val == NULL) {
                        {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
                    }
                    min_val = PyLong_AsLongLong(py_min_val);
                    Py_DECREF(py_min_val);
                    if (min_val == -1 && PyErr_Occurred()) {
                        {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
                    }
                    PyObject* py_max_val = PyArray_Max(V3, NPY_MAXDIMS,
                                                       NULL);
                    if (py_max_val == NULL) {
                        {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
                    }
                    max_val = PyLong_AsLongLong(py_max_val);
                    Py_DECREF(py_max_val);
                    if (max_val == -1 && PyErr_Occurred()) {
                        {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
                    }
                    if (min_val < NPY_MIN_INTP || max_val > NPY_MAX_INTP) {
                        PyErr_SetString(PyExc_IndexError,
                                     "Index contains values "
                                     "that are bigger than the maximum array "
                                     "size on this system.");
                        {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
                    }
                }
                indices = (PyArrayObject*) PyArray_Cast(V3, NPY_INTP);
                if (indices == NULL) {
                    {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
                }
            }
            else {
                 indices = V3;
                 Py_INCREF(indices);
            }
            if (V1 != NULL) {
                npy_intp nd, i, *shape;
                nd = PyArray_NDIM(V5) + PyArray_NDIM(indices) - 1;
                if (PyArray_NDIM(V1) != nd) {
                    Py_CLEAR(V1);
                }
                else {
                    shape = PyArray_DIMS(V1);
                    for (i = 0; i < PyArray_NDIM(indices); i++) {
                        if (shape[i] != PyArray_DIMS(indices)[i]) {
                            Py_CLEAR(V1);
                            break;
                        }
                    }
                    if (V1 != NULL) {
                        for (; i < nd; i++) {
                            if (shape[i] != PyArray_DIMS(V5)[
                                                i-PyArray_NDIM(indices)+1]) {
                                Py_CLEAR(V1);
                                break;
                            }
                        }
                    }
                }
            }
            V1 = (PyArrayObject*)PyArray_TakeFrom(
                        V5, (PyObject*)indices, 0, V1, NPY_RAISE);
            Py_DECREF(indices);
            if (V1 == NULL) {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_7;};
        __label_7:

double __DUMMY_7;

}
__label_6:

        if (V5) {
            Py_XDECREF(V5);
        }
        
    {Py_XDECREF(py_V5);}
    
double __DUMMY_6;

}
__label_4:

        if (V3) {
            Py_XDECREF(V3);
        }
        
    {Py_XDECREF(py_V3);}
    
double __DUMMY_4;

}
__label_2:

    if (!__failure) {
      
        {Py_XDECREF(py_V1);}
        if (!V1) {
            Py_INCREF(Py_None);
            py_V1 = Py_None;
        }
        else if ((void*)py_V1 != (void*)V1) {
            py_V1 = (PyObject*)V1;
        }

        {Py_XINCREF(py_V1);}

        if (V1 && !PyArray_ISALIGNED((PyArrayObject*) py_V1)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V1),
                         (long int) PyArray_NDIM(V1),
                         (long int) (PyArray_NDIM(V1) >= 3 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-3] : -1),
                         (long int) (PyArray_NDIM(V1) >= 2 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-2] : -1),
                         (long int) (PyArray_NDIM(V1) >= 1 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-1] : -1),
                         (long int) (PyArray_NDIM(V1) >= 3 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-3] : -1),
                         (long int) (PyArray_NDIM(V1) >= 2 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-2] : -1),
                         (long int) (PyArray_NDIM(V1) >= 1 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-1] : -1)
        );
            {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V1, 0);
      {Py_XINCREF(py_V1);}
      PyList_SET_ITEM(storage_V1, 0, py_V1);
      {Py_XDECREF(old);}
    }
    
        if (V1) {
            Py_XDECREF(V1);
        }
        
    {Py_XDECREF(py_V1);}
    
double __DUMMY_2;

}

            
        if (__failure) {
            // When there is a failure, this code puts the exception
            // in __ERROR.
            PyObject* err_type = NULL;
            PyObject* err_msg = NULL;
            PyObject* err_traceback = NULL;
            PyErr_Fetch(&err_type, &err_msg, &err_traceback);
            if (!err_type) {err_type = Py_None;Py_INCREF(Py_None);}
            if (!err_msg) {err_msg = Py_None; Py_INCREF(Py_None);}
            if (!err_traceback) {err_traceback = Py_None; Py_INCREF(Py_None);}
            PyObject* old_err_type = PyList_GET_ITEM(__ERROR, 0);
            PyObject* old_err_msg = PyList_GET_ITEM(__ERROR, 1);
            PyObject* old_err_traceback = PyList_GET_ITEM(__ERROR, 2);
            PyList_SET_ITEM(__ERROR, 0, err_type);
            PyList_SET_ITEM(__ERROR, 1, err_msg);
            PyList_SET_ITEM(__ERROR, 2, err_traceback);
            {Py_XDECREF(old_err_type);}
            {Py_XDECREF(old_err_msg);}
            {Py_XDECREF(old_err_traceback);}
        }
        // The failure code is returned to index what code block failed.
        return __failure;
        
        }
    };
    }
    

        static int __struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49_executor(__struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49 *self) {
            return self->run();
        }

        static void __struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49_destructor(PyObject *capsule) {
            __struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49 *self = (__struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49 *)PyCapsule_GetContext(capsule);
            delete self;
        }
    
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (4 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 4, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49* struct_ptr = new __struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
    PyObject* thunk = PyCapsule_New((void*)(&__struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49_executor), NULL, __struct_compiled_op_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49_destructor);
    if (thunk != NULL && PyCapsule_SetContext(thunk, struct_ptr) != 0) {
        PyErr_Clear();
        Py_DECREF(thunk);
        thunk = NULL;
    }

  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49",
  NULL,
  -1,
  MyMethods,
};

PyMODINIT_FUNC PyInit_m9502af10e1a2172b4ebfa71cdb12fb12ff0181f2f5502719731b5113680c9d49(void) {
   import_array();
   
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
