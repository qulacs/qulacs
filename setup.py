import os
import re
import sys
import platform
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

_VERSION = '0.2.0'

project_name = 'Qulacs'

def _is_valid_compiler(cmd):
    try:
        out = subprocess.check_output([cmd, '-dumpfullversion', '-dumpversion']).decode()
        version = LooseVersion(out)
        return version >= LooseVersion('7.0.0')
    except:
        return False

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ('opt-flags=', 'o', 'optimization flags for compiler')
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.opt_flags = None

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DPYTHON_SETUP_FLAG:STR=Yes']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            cmake_args += ['-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            env_gcc = os.getenv('C_COMPILER')
            if env_gcc:
                gcc_candidates = [env_gcc]
            else:
                gcc_candidates = ['gcc', 'gcc-9', 'gcc-8', 'gcc-7']
            gcc = next(iter(filter(_is_valid_compiler, gcc_candidates)), None)

            env_gxx = os.getenv('CXX_COMPILER')
            if env_gxx:
                gxx_candidates = [env_gxx]
            else:
                gxx_candidates = ['g++', 'g++-9', 'g++-8', 'g++-7']
            gxx = next(iter(filter(_is_valid_compiler, gxx_candidates)), None)

            if gcc is None or gxx is None:
                raise RuntimeError("gcc/g++ >= 7.0.0 must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

            cmake_args += ['-DCMAKE_C_COMPILER=' + gcc]
            cmake_args += ['-DCMAKE_CXX_COMPILER=' + gxx]

            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        if self.opt_flags is not None:
            opt_flags = self.opt_flags
        elif os.getenv('QULACS_OPT_FLAGS'):
            opt_flags = os.getenv('QULACS_OPT_FLAGS')
        else:
            opt_flags = None
        if opt_flags:
            cmake_args += ['-DOPT_FLAGS=' + opt_flags]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'python'] + build_args, cwd=self.build_temp)

setup(
    name=project_name,
    version=_VERSION,
    author='QunaSys',
    author_email='qulacs@qunasys.com',
    url='http://www.qulacs.org',
    description='Quantum circuit simulator for research',
    long_description='',
    packages=find_packages(exclude=['test*']),
    include_package_data=True,
    ext_modules=[CMakeExtension('qulacs')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    test_suite = 'test',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Communications :: Email',
    ],
)
