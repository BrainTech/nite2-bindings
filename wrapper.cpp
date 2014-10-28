//---------------------------
// OpenNI 2 + NiTE 2 wrapper
//---------------------------

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API

#include "boost_python.h"
#include "openni2.h"
#include "nite2.h"

#define VERSION_MAJOR     0
#define VERSION_MINOR     5
#define VERSION_REVISION  2

#define STRINGIFY(s) #s
#define TOSTRING(s) STRINGIFY(s)
#define VERSION_STRING TOSTRING(VERSION_MAJOR) "." TOSTRING(VERSION_MINOR) "." TOSTRING(VERSION_REVISION)

BOOST_PYTHON_MODULE(nite2)
{
    import_array(); // initialize numpy

    bp::numeric::array::set_module_and_type("numpy", "ndarray");

    // Customize what will appear on the docstrings:
    //   Custom docstring -> ON
    //   Python signature -> ON
    //   C++ signature    -> OFF
    bp::docstring_options local_docstring_options(true, true, false);

    bp::scope().attr("__doc__") = "OpenNI 2 + NiTE 2 wrapper";
    bp::scope().attr("__version__") = VERSION_STRING;
    bp::scope().attr("__version_info__") = bp::make_tuple(VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION);

    export_openni2();
    export_nite2();
}
