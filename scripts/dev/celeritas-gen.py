#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""\
Generate file stubs for Celeritas.
"""

from datetime import datetime
import os
import os.path
import re
import subprocess
import stat
import sys

###############################################################################

CLIKE_TOP = '''\
//{modeline:-^75s}//
// Copyright {year} UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \\file {filename}
//---------------------------------------------------------------------------//
'''

HEADER_FILE = '''\
#pragma once

{namespace_begin}
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \\code
    {name} ...;
   \\endcode
 */
class {name}
{{
  public:
    //!@{{
    //! Type aliases
    <++>
    //!@}}

  public:
    // Construct with defaults
    inline {name}();
}};

//---------------------------------------------------------------------------//
{namespace_end}

#include "{name}.i.{hext}"
'''

INLINE_FILE = '''\

{namespace_begin}
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
{name}::{name}()
{{
}}

//---------------------------------------------------------------------------//
{namespace_end}
'''

CODE_FILE = '''\
#include "{name}.{hext}"

{namespace_begin}
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
{namespace_end}
'''

TEST_HARNESS_FILE = '''\
#include "{dirfromtest}/{name}.{hext}"

#include "celeritas_test.hh"
// #include "{name}.test.hh"

using {namespace}::{name};
// using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class {name}Test : public celeritas::Test
{{
  protected:
    void SetUp() override {{}}
}};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F({name}Test, all)
{{
    // {capabbr}TestInput input;
    // input.num_threads = 0;
    // auto result = {lowabbr}_test(input);
    // PRINT_EXPECTED(result.foo);
    // EXPECT_VEC_SOFT_EQ(expected_foo, result.foo);
}}
'''

TEST_HEADER_FILE = '''
namespace celeritas_test
{{
using celeritas::Ownership;
using celeritas::MemSpace;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct {capabbr}TestInput
{{
    int num_threads;
}};

//---------------------------------------------------------------------------//
//! Output results
struct {capabbr}TestOutput
{{
}};

//---------------------------------------------------------------------------//
//! Run on device and return results
{capabbr}TestOutput {lowabbr}_test({capabbr}TestInput);

//---------------------------------------------------------------------------//
}} // namespace celeritas_test
'''

TEST_CODE_FILE = '''\
#include "{name}.test.hh"

#include <thrust/device_vector.h>
#include "base/KernelParamCalculator.cuda.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{{
namespace
{{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void {lowabbr}_test_kernel(unsigned int size)
{{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= size)
        return;
}}
}}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
{capabbr}TestOutput {lowabbr}_test({capabbr}TestInput input)
{{
    static const celeritas::KernelParamCalculator calc_launch_params(
        {lowabbr}_test_kernel, "{lowabbr}_test");
    auto params = calc_launch_params(input.num_threads);
    {lowabbr}_test_kernel<<<params.grid_size, params.block_size>>>(
        input.num_threads);

    CELER_CUDA_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    {capabbr}TestOutput result;
    return result;
}}

//---------------------------------------------------------------------------//
}} // namespace celeritas_test
'''


SWIG_FILE = '''\
%{{
#include "{name}.{hext}"
%}}

%include "{name}.{hext}"
'''


CMAKE_TOP = '''\
#{modeline:-^77s}#
# Copyright {year} UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
'''

CMAKELISTS_FILE = '''\
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
'''


CMAKE_FILE = '''\
#[=======================================================================[.rst:

{name}
-------------------

Description of overall module contents goes here.

.. command:: my_command_name

  Pass the given compiler-dependent warning flags to a library target::

    my_command_name(<target>
                    <INTERFACE|PUBLIC|PRIVATE>
                    LANGUAGE <lang> [<lang>...]
                    [CACHE_VARIABLE <name>])

  ``target``
    Name of the library target.

  ``scope``
    One of ``INTERFACE``, ``PUBLIC``, or ``PRIVATE``. ...

#]=======================================================================]

function(my_command_name)
endfunction()

#-----------------------------------------------------------------------------#
'''

PYTHON_TOP = '''\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright {year} UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
'''

PYTHON_FILE = '''\
"""
"""

'''

SHELL_TOP = '''\
#!/bin/sh -ex
# Copyright {year} UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
'''

SHELL_FILE = '''\

'''

OMN_TOP = '''\
! Copyright {year} UT-Battelle, LLC, and other Celeritas developers.
! See the top-level COPYRIGHT file for details.
! SPDX-License-Identifier: (Apache-2.0 OR MIT)
'''

ORANGE_FILE = '''
[GEOMETRY]
global "global"
comp         : matid
    galactic   0
    detector   1

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!

[UNIVERSE=general global]
interior "world_box"

[UNIVERSE][SHAPE=box world_box]
widths 10000 10000 10000  ! note: units are in cm

[UNIVERSE][SHAPE=cyl mycyl]
axis z
radius 10
length 20

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!

[UNIVERSE][CELL detector]
comp detector
shapes mycyl

[UNIVERSE][CELL world_fill]
comp galactic
shapes world_box ~mycyl
'''

YEAR = datetime.today().year

TEMPLATES = {
    'hh': HEADER_FILE,
    'i.hh': INLINE_FILE,
    'cc': CODE_FILE,
    'cu': CODE_FILE,
    'cuh': HEADER_FILE,
    'test.cc': TEST_HARNESS_FILE,
    'test.cu': TEST_CODE_FILE,
    'test.hh': TEST_HEADER_FILE,
    'k.cuh': INLINE_FILE,
    'i.cuh': INLINE_FILE,
    't.cuh': INLINE_FILE,
    'i': SWIG_FILE,
    'cmake': CMAKE_FILE,
    'CMakeLists.txt': CMAKELISTS_FILE,
    'py': PYTHON_FILE,
    'sh': SHELL_FILE,
    'org.omn': ORANGE_FILE,
}

LANG = {
    'h': "C++",
    'hh': "C++",
    'cc': "C++",
    'cu': "CUDA",
    'cuh': "CUDA",
    'cmake': "CMake",
    'i': "SWIG",
    'CMakeLists.txt': "CMake",
    'py': "Python",
    'sh': "Shell",
    'omn': "Omnibus",
}

TOPS = {
    'C++': CLIKE_TOP,
    'CUDA': CLIKE_TOP,
    'SWIG': CLIKE_TOP,
    'CMake': CMAKE_TOP,
    'Python': PYTHON_TOP,
    'Shell': SHELL_TOP,
    'Omnibus': OMN_TOP,
}

HEXT = {
    'C++': "hh",
    'CUDA': "cuh",
    'SWIG': "hh",
}


def generate(root, filename, namespace):
    if os.path.exists(filename):
        print("Skipping existing file " + filename)
        return

    relpath = os.path.relpath(filename, start=root)
    dirname = os.path.dirname(relpath)
    basename = os.path.basename(filename)
    (name, _, longext) = basename.partition('.')

    lang = None
    template = None
    ext = longext.split('.')[-1]
    for check_lang in [basename, longext, ext]:
        if not lang:
            lang = LANG.get(check_lang, None)
        if not template:
            template = TEMPLATES.get(check_lang, None)
    if not lang:
        print(f"No known language for '.{ext}' files")
    if not template:
        print(f"No configured template for '.{ext}' files")
    if not lang or not template:
        sys.exit(1)

    top = TOPS[lang]
    nsbeg = []
    nsend = []
    for subns in namespace.split('::'):
        nsbeg.append(f'namespace {subns}\n{{')
        nsend.append(f'}} // namespace {subns}')

    relpath = re.sub(r'^[./]+', '', relpath)
    capabbr = re.sub(r'[^A-Z]+', '', name)
    dirfromtest = re.sub(r'^test/', '', os.path.dirname(relpath))
    variables = {
        'longext': longext,
        'ext': ext,
        'hext': HEXT.get(lang, ext),
        'modeline': "-*-{}-*-".format(lang),
        'name': name,
        'namespace': namespace,
        'namespace_begin': "\n".join(nsbeg),
        'namespace_end': "\n".join(reversed(nsend)),
        'filename': filename,
        'basename': basename,
        'dirfromtest': dirfromtest,
        'capabbr': capabbr,
        'lowabbr': capabbr.lower(),
        'year': YEAR,
    }
    with open(filename, 'w') as f:
        f.write((top + template).format(**variables))
        if top.startswith('#!'):
            # Set executable bits
            mode = os.fstat(f.fileno()).st_mode
            mode |= 0o111
            os.fchmod(f.fileno(), stat.S_IMODE(mode))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'filename', nargs='+',
        help='file names to generate')
    parser.add_argument(
        '--basedir',
        help='root source directory for file naming')
    parser.add_argument(
        '--namespace', '-n',
        default='celeritas',
        help='root source directory for file naming')
    args = parser.parse_args()
    basedir = args.basedir or os.path.join(
        subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
        .decode().strip(),
        'src')
    for fn in args.filename:
        generate(basedir, fn, args.namespace)


if __name__ == '__main__':
    main()
