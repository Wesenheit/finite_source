#include <Python.h>
#include "numpy/arrayobject.h"
#include "finite.h"

/* This module provides an interface for calculating microlensing 
 * magnifications for extended sources using the method proposed by
 * Bozza et al. 2018.
 * 
 * See for extending Python using C: 
 *      https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
 *      https://docs.python.org/2/extending/extending.html
 *      https://dfm.io/posts/python-c-extensions/
 * 
 * P. Mroz @ IPAC, 30 Apr 2019
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

static char func_docstring1[] = 
    "Extended source magnification for a uniform source.";
    
static char func_docstring2[] = 
    "Extended source magnification for a limb-darkened source.";

static PyObject *
ampl_wrapper (PyObject *self, PyObject *args)
{
    
    /* This is Python wrapper for ampl(u,rho) */
    
    PyObject *func0_obj,*func1_obj; 
    /* pointers to PyObject objects containing pre-calculated magnifications */
    double *func0,*func1; /* pointers to arrays containing the data */
    double u,rho,A;
    
    /* Parsing arguments */
    if (!PyArg_ParseTuple(args,"ddOO",&u,&rho,&func0_obj,&func1_obj)) return NULL;
    
    /* Interpret the input objects as numpy arrays */
    PyObject *func0_ = PyArray_FROM_OTF(func0_obj,NPY_DOUBLE,NPY_IN_ARRAY);
    PyObject *func1_ = PyArray_FROM_OTF(func1_obj,NPY_DOUBLE,NPY_IN_ARRAY);
    
    /* If that didn't work, throw an exception */
    if ( func0_ == NULL || func1_ == NULL ) {
        Py_XDECREF(func0_);
        Py_XDECREF(func1_);
        return NULL;
    }

    /* Get pointers to the data as C-types */
    func0 = (double *) PyArray_DATA(func0_);
    func1 = (double *) PyArray_DATA(func1_);
        
    /* Call the external C function to calculate the magnification */
    A = ampl(u,rho,func0,func1);
    
    /* Cleaning-up */
    Py_DECREF(func0_);
    Py_DECREF(func1_);
    
    /* Return the calculated value */
    return Py_BuildValue("d",A);
}

static PyObject *
ampl_ld_wrapper (PyObject *self, PyObject *args)
{
    
    /* This is Python wrapper for ampl_ld(u,rho,Gamma,Lambda) */
    
    PyObject *func0_obj,*func1_obj; 
    /* pointers to PyObject objects containing pre-calculated magnifications */
    double *func0,*func1; /* pointers to arrays containing the data */
    double u,rho,Gamma,Lambda,A;
    int n_rings;
    
    /* Parsing arguments */
    if (!PyArg_ParseTuple(args,"ddddOOi",&u,&rho,&Gamma,&Lambda,
        &func0_obj,&func1_obj,&n_rings)) return NULL;

    /* Interpret the input objects as numpy arrays */
    PyObject *func0_ = PyArray_FROM_OTF(func0_obj,NPY_DOUBLE,NPY_IN_ARRAY);
    PyObject *func1_ = PyArray_FROM_OTF(func1_obj,NPY_DOUBLE,NPY_IN_ARRAY);
    
    /* If that didn't work, throw an exception */
    if ( func0_ == NULL || func1_ == NULL ) {
        Py_XDECREF(func0_);
        Py_XDECREF(func1_);
        return NULL;
    }

    /* Get pointers to the data as C-types */
    func0 = (double *) PyArray_DATA(func0_);
    func1 = (double *) PyArray_DATA(func1_);
        
    /* Call the external C function to calculate the magnification */
    A = ampl_ld(u,rho,Gamma,Lambda,func0,func1,n_rings);

    /* Cleaning-up */
    Py_DECREF(func0_);
    Py_DECREF(func1_);
    
    /* Return the calculated value */
    return Py_BuildValue("d",A);
}

static PyMethodDef FiniteMethods[] = {
    {"ampl_ld", (PyCFunction) ampl_ld_wrapper, METH_VARARGS, func_docstring2},
    {"ampl", (PyCFunction) ampl_wrapper, METH_VARARGS, func_docstring1},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initFinite(void)
{
    (void)Py_InitModule("Finite", FiniteMethods);
    
    /* Load numpy functionality.*/
    import_array();
}
