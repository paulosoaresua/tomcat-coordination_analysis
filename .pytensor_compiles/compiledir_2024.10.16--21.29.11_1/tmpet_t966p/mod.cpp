#include <Python.h>
#include <iostream>
#include "pytensor_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
//////////////////////
////  Support Code
//////////////////////

    namespace {
    struct __struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2 {
        PyObject* __ERROR;

        PyObject* storage_V5;
PyObject* storage_V3;
PyObject* storage_V1;
        

        __struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2() {
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
        ~__struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2(void) {
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
    
                typedef npy_int64 dtype_V1;
            
        npy_int64 V1;
        
    PyObject* py_V3;
    
                typedef npy_int64 dtype_V3;
            
        npy_int64 V3;
        
    PyObject* py_V5;
    
                typedef npy_int64 dtype_V5;
            
        npy_int64 V5;
        
{

    py_V1 = PyList_GET_ITEM(storage_V1, 0);
    {Py_XINCREF(py_V1);}
    
        if (py_V1 == Py_None)
        {
            
        V1 = 0;
        
        }
        else
        {
            
            if (!PyObject_TypeCheck(py_V1, &PyInt64ArrType_Type))
            {
                PyErr_Format(PyExc_ValueError,
                    "Scalar check failed (npy_int64)");
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            
        PyArray_ScalarAsCtype(py_V1, &V1);
        
        }
        
{

    py_V3 = PyList_GET_ITEM(storage_V3, 0);
    {Py_XINCREF(py_V3);}
    
            if (!PyObject_TypeCheck(py_V3, &PyInt64ArrType_Type))
            {
                PyErr_Format(PyExc_ValueError,
                    "Scalar check failed (npy_int64)");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            
        PyArray_ScalarAsCtype(py_V3, &V3);
        
{

    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
            if (!PyObject_TypeCheck(py_V5, &PyInt64ArrType_Type))
            {
                PyErr_Format(PyExc_ValueError,
                    "Scalar check failed (npy_int64)");
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            
        PyArray_ScalarAsCtype(py_V5, &V5);
        
{
// Op class Sub
V1 = V5 - V3;__label_7:

double __DUMMY_7;

}
__label_6:

    {Py_XDECREF(py_V5);}
    
double __DUMMY_6;

}
__label_4:

    {Py_XDECREF(py_V3);}
    
double __DUMMY_4;

}
__label_2:

    if (!__failure) {
      
        Py_XDECREF(py_V1);
        py_V1 = PyArrayScalar_New(Int64);
        if (!py_V1)
        {
            Py_XINCREF(Py_None);
            py_V1 = Py_None;
            PyErr_Format(PyExc_MemoryError,
                "Instantiation of new Python scalar failed (npy_int64)");
            {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
        }
        PyArrayScalar_ASSIGN(py_V1, Int64, V1);
        
      PyObject* old = PyList_GET_ITEM(storage_V1, 0);
      {Py_XINCREF(py_V1);}
      PyList_SET_ITEM(storage_V1, 0, py_V1);
      {Py_XDECREF(old);}
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
    

        static int __struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2_executor(__struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2 *self) {
            return self->run();
        }

        static void __struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2_destructor(PyObject *capsule) {
            __struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2 *self = (__struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2 *)PyCapsule_GetContext(capsule);
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
  __struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2* struct_ptr = new __struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
    PyObject* thunk = PyCapsule_New((void*)(&__struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2_executor), NULL, __struct_compiled_op_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2_destructor);
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
  "m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2",
  NULL,
  -1,
  MyMethods,
};

PyMODINIT_FUNC PyInit_m7b1ec301b8727798be69251971d2367efa642ae54bf6c683ed146d8d2f0d56b2(void) {
   import_array();
   
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
