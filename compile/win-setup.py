from distutils.extension import Extension
from distutils.core import setup
from Cython.Distutils import build_ext
import numpy as np

numpy_include = np.get_include()
ext_modules = [
Extension(
        "install.core.utils.cython_bbox",
        ["bbox_normalized.pyx"],
        include_dirs = [numpy_include]
    ),
    Extension(
        "install.core.nms.cpu_nms",
        ["cpu_nms.pyx"],
        include_dirs = [numpy_include]
    ),
]

setup(name='ssd',ext_modules=ext_modules,cmdclass = {'build_ext': build_ext})
