from setuptools import setup, Extension
import pybind11

# Define the C++ extension module
ext_modules = [
    Extension(
        'fast_router',                     # The name of the module you will import in Python
        ['router_math.cpp'],               # The C++ source file
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['/O2', '/fp:fast']  # MSVC flags: Max speed + Fast floating point math
    ),
]

setup(
    name='fast_router',
    ext_modules=ext_modules,
)