#ifndef BOOST_PYTHON_LOCAL_H
#define BOOST_PYTHON_LOCAL_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>

// run: numpy.get_include() to get include path for numpy
// usually: 'C:\Python27\lib\site-packages\numpy\core\include'

#include <numpy/arrayobject.h>

namespace bp = boost::python;

#endif // BOOST_PYTHON_LOCAL_H
