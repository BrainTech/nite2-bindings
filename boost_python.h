#ifndef BOOST_PYTHON_LOCAL_H
#define BOOST_PYTHON_LOCAL_H

#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>

// run: numpy.get_include() to get include path for numpy
// usually: 'C:\Python27\lib\site-packages\numpy\core\include'

#include <numpy/arrayobject.h>

namespace bp = boost::python;

#endif // BOOST_PYTHON_LOCAL_H
