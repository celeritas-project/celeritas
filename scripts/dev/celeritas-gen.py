#!/usr/bin/env python3
# Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
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

CXX_TOP = '''\
//{modeline:-^75s}//
// Copyright {year} UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \\file {dirname}{basename}
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
    //! \\name Type aliases
    <++>
    //!@}}

  public:
    // Construct with defaults
    inline {name}();
}};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
#include "{dirname}{name}.{hext}"

#include "celeritas_test.hh"
// #include "{name}.test.hh"

{namespace_begin}
//---------------------------------------------------------------------------//

class {name}Test : public ::celeritas::test::Test
{{
  protected:
    void SetUp() override {{}}
}};

TEST_F({name}Test, host)
{{
    // PRINT_EXPECTED(result.foo);
    // EXPECT_VEC_SOFT_EQ(expected_foo, result.foo);
}}

// TEST_F({name}Test, TEST_IF_CELER_DEVICE(device))
// {{
//     {capabbr}TestInput input;
//     {lowabbr}_test(input);
// }}

//---------------------------------------------------------------------------//
{namespace_end}
'''

TEST_HEADER_FILE = '''
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

{namespace_begin}
//---------------------------------------------------------------------------//
// DATA
//---------------------------------------------------------------------------//
template<Ownership W, MemSpace M>
struct {capabbr}TestParamsData
{{
    // FIXME
    // {capabbr}ParamsData<W, M>  geometry;
    // RngParamsData<W, M>  rng;

    explicit CELER_FUNCTION operator bool() const
    {{
        // FIXME
        // return geometry && rng;
        return false;
    }}

    template<Ownership W2, MemSpace M2>
    {capabbr}TestParamsData& operator=(const {capabbr}TestParamsData<W2, M2>& other)
    {{
        CELER_EXPECT(other);
        // FIXME
        // geometry = other.geometry;
        // rng      = other.rng;
        return *this;
    }}
}};

//---------------------------------------------------------------------------//
template<Ownership W, MemSpace M>
struct {capabbr}TestStateData
{{
    template<class T>
    using StateItems = {corecel_ns}StateCollection<T, W, M>;

    // FIXME
    // {capabbr}StateData<W, M> geometry;
    // RngStateData<W, M> rng;
    // StateItems<bool> alive;

    CELER_FUNCTION {corecel_ns}size_type size() const {{
        // FIXME
       // return geometry.size();
       }}

    explicit CELER_FUNCTION operator bool() const
    {{
        // FIXME
        // return geometry && rng && !alive.empty();
        return false;
    }}

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    {capabbr}TestStateData& operator=({capabbr}TestStateData<W2, M2>& other)
    {{
        CELER_EXPECT(other);
        // FIXME
        // geometry = other.geometry;
        // rng      = other.rng;
        // alive    = other.alive;
        return *this;
    }}
}};

//---------------------------------------------------------------------------//
template<MemSpace M>
inline void resize({capabbr}TestStateData<Ownership::value, M>* state,
                   const HostCRef<{capabbr}TestParamsData>&     params,
                   {corecel_ns}size_type                             size)
{{
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    // FIXME
    // resize(&state->geometry, params.geometry, size);
    // resize(&state->alive, size);
    // fill(state->alive, 0)
    CELER_ENSURE(state.size() == size);
}}

//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
struct {capabbr}TestExecutor
{{
    using ParamsRef = NativeCRef<{capabbr}TestParamsData>;
    using StateRef  = NativeRef<{capabbr}TestStateData>;

    const ParamsRef& params;
    const StateRef&  state;

    inline CELER_FUNCTION void operator()({corecel_ns}ThreadId tid) const;
}};

//---------------------------------------------------------------------------//
CELER_FUNCTION void {capabbr}TestExecutor::operator()({corecel_ns}ThreadId tid) const
{{
    // FIXME
}}

//---------------------------------------------------------------------------//
// DEVICE KERNEL EXECUTION
//---------------------------------------------------------------------------//
//! Run on device
void {lowabbr}_test(const DeviceCRef<{capabbr}TestParamsData>&,
            const DeviceRef<{capabbr}TestStateData>&);

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline void {lowabbr}_test(
    const DeviceCRef<{capabbr}TestParamsData>&,
    const DeviceRef<{capabbr}TestStateData>&)
{{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}}
#endif

//---------------------------------------------------------------------------//
{namespace_end}
'''

TEST_CODE_FILE = '''\
#include "{name}.test.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Types.h"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"

{namespace_begin}
namespace
{{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void {lowabbr}_test_kernel(
    const {corecel_ns}DeviceCRef<{capabbr}TestParamsData> params,
    const {corecel_ns}DeviceRef<{capabbr}TestStateData> state)
{{
    auto tid = {corecel_ns}KernelParamCalculator::thread_id();
    if (tid.get() >= state.size())
        return;

    {capabbr}TestExecutor execute{{params, state}};
    execute(tid);
}}
}}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void {lowabbr}_test(
    const {corecel_ns}DeviceCRef<{capabbr}TestParamsData>& params,
    const {corecel_ns}DeviceRef<{capabbr}TestStateData>& state)
{{
    CELER_LAUNCH_KERNEL({lowabbr}_test,
                        {corecel_ns}device().default_block_size(),
                        state.size(),
                        params.ref<MemSpace::native>(),
                        state);

    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}}

//---------------------------------------------------------------------------//
{namespace_end}
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

RST_TOP = '''\
.. Copyright {year} UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0
'''

RST_FILE = '''
.. _{name}:

****************
{name}
****************

Text with a link to `Sphinx primer`_ and `RST`_ docs.

.. _Sphinx primer : https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _RST : https://docutils.sourceforge.io/docs/user/rst/quickref.html

Subsection
==========

Another paragraph.

.. note:: Don't start a subsection immediately after a section: make sure
   there's something to say at the start of each one.

Subsubsection
-------------

These are useful for heavily nested documentation such as API descriptions. ::

    // This code block will be highlighted in the default language, which for
    // Celeritas is C++.
    int i = 0;
'''

YEAR = datetime.today().year

TEMPLATES = {
    'hh': HEADER_FILE,
    'cc': CODE_FILE,
    'cu': CODE_FILE,
    'test.cc': TEST_HARNESS_FILE,
    'test.cu': TEST_CODE_FILE,
    'test.hh': TEST_HEADER_FILE,
    'i': SWIG_FILE,
    'cmake': CMAKE_FILE,
    'CMakeLists.txt': CMAKELISTS_FILE,
    'py': PYTHON_FILE,
    'sh': SHELL_FILE,
    'org.omn': ORANGE_FILE,
    'rst': RST_FILE,
}

LANG = {
    'h': "C",
    'hh': "C++",
    'cc': "C++",
    'cu': "CUDA",
    'cmake': "CMake",
    'i': "SWIG",
    'CMakeLists.txt': "CMake",
    'py': "Python",
    'sh': "Shell",
    'omn': "Omnibus",
    'rst': "RST",
}

TOPS = {
    'C': CXX_TOP,
    'C++': CXX_TOP,
    'CUDA': CXX_TOP,
    'SWIG': CXX_TOP,
    'CMake': CMAKE_TOP,
    'Python': PYTHON_TOP,
    'Shell': SHELL_TOP,
    'Omnibus': OMN_TOP,
    'RST': RST_TOP,
}

HEXT = {
    'C': "h",
    'C++': "hh",
    'CUDA': "hh",
    'SWIG': "hh",
}


def generate(repodir, filename, namespace):
    if os.path.exists(filename):
        print("Skipping existing file " + filename)
        return

    dirname = os.path.relpath(filename, start=repodir)
    all_dirs = dirname.split(os.sep)[:-1]

    if namespace is None:
        namespace = 'celeritas'
        if all_dirs[0] == 'app':
            namespace += '::app'
        elif all_dirs[0] == 'test':
            namespace += '::test'
        if all_dirs[-1] == 'detail':
            namespace += '::detail'

    # Construct directory name with src/app/test dropped
    dirname = os.sep.join(all_dirs[1:])
    if dirname:
        dirname += os.sep

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
        nsend.append(f'}}  // namespace {subns}')

    capabbr = re.sub(r'[^A-Z]+', '', name)
    variables = {
        'longext': longext,
        'ext': ext,
        'hext': "hh",
        'modeline': "-*-{}-*-".format(lang),
        'name': name,
        'namespace': namespace,
        'namespace_begin': "\n".join(nsbeg),
        'namespace_end': "\n".join(reversed(nsend)),
        'basename': basename,
        'dirname': dirname,
        'capabbr': capabbr,
        'lowabbr': capabbr.lower(),
        'year': YEAR,
        'corecel_ns': "", # or "celeritas::" or someday "corecel::"
        'celeritas_ns': "",
    }
    with open(filename, 'w') as f:
        f.write((top + template).format(**variables))
        if top.startswith('#!'):
            # Set executable bits
            mode = os.fstat(f.fileno()).st_mode
            mode |= 0o111
            os.fchmod(f.fileno(), stat.S_IMODE(mode))
    return filename


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'filename', nargs='+',
        help='file names to generate')
    parser.add_argument(
        '--repodir',
        help='root source directory for file naming')
    parser.add_argument('-o', '--open',
        action='store_true',
        help='call "open" on the created files')
    parser.add_argument(
        '--namespace', '-n',
        default=None,
        help='C++ namespace to generate')
    args = parser.parse_args()
    repodir = args.repodir or (
        subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
        .decode().strip()
    )
    generated = []
    for fn in args.filename:
        fn = generate(repodir, fn, args.namespace)
        if fn:
            generated.append(fn)

    if args.open and generated:
        subprocess.call(["open"] + generated)


if __name__ == '__main__':
    main()
