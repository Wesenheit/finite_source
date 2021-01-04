from distutils.core import setup, Extension
import numpy as np
import sys

# COMPILATION: python setup.py build_ext --inplace

python_version = sys.version_info[0]

if python_version == 3:
    module = Extension(name='Finite', 
                       sources=['source/python_3_wrapper.c','source/finite.c'], 
                       include_dirs=[np.get_include()])
elif python_version == 2:
    module = Extension(name='Finite', 
                       sources=['source/python_wrapper.c','source/finite.c'], 
                       include_dirs=[np.get_include()])
else:
    exit()
    
setup(
    name = 'Finite', 
    ext_modules = [module],
)
