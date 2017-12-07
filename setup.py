from setuptools import setup
import os
import shutil
import sys

__version__ = '0.0.1'

try:
    import tensorflow as tf  # noqa
    _version = tf.__version__.split('.')
    assert int(_version[0]) >= 1, "TF>=1.0 is required!"
    _HAS_TF = True
except ImportError as e:
    print(e)
    _HAS_TF = False

if not _HAS_TF:
    print('The tensorFlow seems not have been installed, please install it first.')
    exit(1)

# # parse scripts
# scripts = ['scripts/plot-point.py', 'scripts/dump-model-params.py']
# scripts_to_install = []
# for s in scripts:
#     dirname = os.path.dirname(s)
#     basename = os.path.basename(s)
#     if basename.endswith('.py'):
#         basename = basename[:-3]
#     newname = 'tpk-' + basename  # install scripts with a prefix to avoid name confusion
#     # setup.py could be executed the second time in the same dir
#     if not os.path.isfile(newname):
#         shutil.move(s, newname)
#     scripts_to_install.append(newname)

setup(
    name='tensorpack',
    version=__version__,
    description='Low level vision library for TensorFlow',
    install_requires=['numpy'],
    tests_require=['flake8', 'scikit-image'],
)