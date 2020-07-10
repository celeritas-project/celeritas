#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
# File: scripts/dev/celeritas-gen.py
###############################################################################
"""Generate class file stubs for Celeritas.
"""
from __future__ import (division, absolute_import, print_function)
import os.path
import re
import subprocess
import sys
###############################################################################

CLIKE_TOP = '''\
//{modeline:-^75s}//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \\file {filename}
//---------------------------------------------------------------------------//
'''

HEADER_FILE = '''\
#ifndef {header_guard}
#define {header_guard}

namespace celeritas
{{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    {name} ...;
   \endcode
 */
class {name}
{{
  public:
    //@{{
    //! Type aliases
    <++>
    //@}}

  public:
    // Construct with defaults
    inline {name}();
}};

//---------------------------------------------------------------------------//
}} // namespace celeritas

#include "{name}.i.{hext}"

#endif // {header_guard}
'''

INLINE_FILE = '''\

namespace celeritas
{{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
{name}::{name}()
{{
}}

//---------------------------------------------------------------------------//
}}  // namespace celeritas
'''

CODE_FILE = '''\
#include "{name}.{hext}"

namespace celeritas
{{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
}} // namespace celeritas
'''

TEST_HARNESS_FILE = '''\
#include "{dirname}/{name}.{hext}"

#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "gmock/gmock.h"
#include "{name}.test.hh"

using celeritas::{name};
using namespace celeritas_test;

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
    // using testing::ElementsAreArray;
    // EXPECT_THAT(result.foo, ElementsAreArray(expected_foo));
}}
'''

TEST_HEADER_FILE = '''
namespace celeritas_test
{{
using namespace celeritas;
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
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void {lowabbr}_test_kernel(unsigned int size)
{{
    unsigned int local_thread_id
        = celeritas::KernelParamCalculator::thread_id().get();
    if (local_thread_id >= size)
        return;
}}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
{capabbr}TestOutput {lowabbr}_test({capabbr}TestInput input)
{{
    celeritas::KernelParamCalculator calc_launch_params;
    auto params = calc_launch_params(input.num_threads);
    {lowabbr}_test_kernel<<<params.grid_size, params.block_size>>>(
        input.num_threads);

    {capabbr}TestOutput result;
    return result;
}}

//---------------------------------------------------------------------------//
}} // namespace celeritas_test
'''


CMAKE_TOP = '''\
#{modeline:-^77s}#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
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
    'cmake': CMAKE_FILE,
    'CMakeLists.txt': CMAKELISTS_FILE,
}

LANG = {
    'h': "C++",
    'hh': "C++",
    'cc': "C++",
    'cu': "CUDA",
    'cuh': "CUDA",
    'cmake': "CMake",
    'CMakeLists.txt': "CMake",
}

TOPS = {
    'C++': CLIKE_TOP,
    'CUDA': CLIKE_TOP,
    'CMake': CMAKE_TOP,
}

HEXT = {
    'C++': "hh",
    'CUDA': "cuh",
}

def generate(root, filename):
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

    relpath = re.sub(r'^[./]+', '', relpath)
    capabbr = re.sub(r'[^A-Z]+', '', name)
    variables = {
        'longext': longext,
        'ext': ext,
        'hext': HEXT.get(lang, ext),
        'modeline': "-*-{}-*-".format(lang),
        'name': name,
        'header_guard': re.sub(r'\W', '_', relpath),
        'filename': filename,
        'basename': basename,
        'dirname': dirname,
        'capabbr': capabbr,
        'lowabbr': capabbr.lower(),
        }
    with open(filename, 'w') as f:
        f.write((top + template).format(**variables))

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', nargs='+',
                        help='file names to generate')
    parser.add_argument('--basedir',
                        help='root source directory for file naming')
    args = parser.parse_args()
    basedir = args.basedir or os.path.join(
            subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
                .decode().strip(),
            'src')
    for fn in args.filename:
        generate(basedir, fn)

#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    main()

###############################################################################
# end of scripts/dev/celeritas-gen.py
###############################################################################
