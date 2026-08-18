#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Generate file stubs for Celeritas."""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

###############################################################################

CODE_LICENSE = "(Apache-2.0 OR MIT)"
DOC_LICENSE = "CC-BY-4.0"
StringOrLines = str | Iterable[str]


def _make_top(
    comment_prefix: str,
    preamble: StringOrLines | None = None,
    postamble: StringOrLines | None = None,
    *,
    license: str = CODE_LICENSE,
) -> str:
    lines: list[str] = []

    def _append_lines(value: StringOrLines | None) -> None:
        if not value:
            return
        if isinstance(value, str):
            lines.append(value)
            return
        lines.extend(value)

    _append_lines(preamble)
    lines.extend(
        f"{comment_prefix} {line}"
        for line in [
            "Copyright Celeritas contributors: see top-level COPYRIGHT file for details",
            f"SPDX-License-Identifier: {license}",
        ]
    )
    _append_lines(postamble)
    lines.append("")
    return "\n".join(lines)


_C_SEP = "".join(["//", "-" * 75, "//"])  # //-----...---//
CXX_TOP = _make_top(
    "//", "//{modeline:-^75s}//", [_C_SEP, "//! \\file {dirname}{basename}", _C_SEP]
)

HEADER_FILE = """\
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
"""

CODE_FILE = """\
#include "{name}.{hext}"

{namespace_begin}
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
{namespace_end}
"""

C_HEADER_FILE = """\
#pragma once

//---------------------------------------------------------------------------//
"""

C_CODE_FILE = """\
#include "{name}.{hext}"

//---------------------------------------------------------------------------//
"""

TEST_HARNESS_FILE = """\
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
"""

TEST_HEADER_FILE = """
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Config.hh"
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
"""

TEST_CODE_FILE = """\
#include "{name}.test.hh"

#include "corecel/DeviceRuntimeApi.hh"
#include "corecel/Types.hh"
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

    CELER_DEVICE_API_CALL(DeviceSynchronize());
}}

//---------------------------------------------------------------------------//
{namespace_end}
"""


CMAKE_TOP = _make_top("#", "#{modeline:-^77s}#")

CMAKELISTS_FILE = """\
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
"""


CMAKE_FILE = """\
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
"""

PYTHON_TOP = _make_top("#", "#!/usr/bin/env python")

PYTHON_FILE = '''\
"""
"""

'''

SHELL_TOP = _make_top(
    "#", ["#!/bin/sh -ex", "#{modeline:-^77s}#"], "#{:-^77s}#".format("")
)

SHELL_FILE = """\

"""

OMN_TOP = _make_top("!")

ORANGE_FILE = """
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
"""

RST_TOP = _make_top("..", license=DOC_LICENSE)

RST_FILE = """
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
"""

TEMPLATES = {
    "hh": HEADER_FILE,
    "c": C_CODE_FILE,
    "h": C_HEADER_FILE,
    "cc": CODE_FILE,
    "cu": CODE_FILE,
    "test.cc": TEST_HARNESS_FILE,
    "test.cu": TEST_CODE_FILE,
    "test.hh": TEST_HEADER_FILE,
    "cmake": CMAKE_FILE,
    "CMakeLists.txt": CMAKELISTS_FILE,
    "py": PYTHON_FILE,
    "sh": SHELL_FILE,
    "org.omn": ORANGE_FILE,
    "rst": RST_FILE,
}

LANG = {
    "h": "C",
    "c": "C",
    "hh": "C++",
    "cc": "C++",
    "cu": "cuda",
    "cmake": "cmake",
    "CMakeLists.txt": "cmake",
    "py": "python",
    "sh": "sh",
    "omn": "omnibus",
    "rst": "rst",
}

TOPS = {
    "C": CXX_TOP,
    "C++": CXX_TOP,
    "cuda": CXX_TOP,
    "cmake": CMAKE_TOP,
    "python": PYTHON_TOP,
    "sh": SHELL_TOP,
    "omnibus": OMN_TOP,
    "rst": RST_TOP,
}

HEXT = {
    "C": "h",
    "C++": "hh",
    "cuda": "hh",
}


def generate(
    repodir: str | Path, filename: str | Path, namespace: str | None
) -> str | None:
    path = Path(filename)
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.exists():
        print(f"Skipping existing file {path}")
        return None

    repo_root = Path(repodir).resolve()
    dirname = Path(os.path.relpath(path, start=str(repo_root)))
    all_dirs = list(dirname.parts[:-1])
    if not all_dirs:
        print("warning: not inside a celeritas subdirectory")
        all_dirs = [""]

    namespace_value = namespace
    if namespace_value is None:
        namespace_value = "celeritas"
        if all_dirs[0] in ("app", "test", "example"):
            namespace_value += "::" + all_dirs[0]
        if all_dirs[-1] == "detail":
            namespace_value += "::detail"

    dirname_str = "/".join(all_dirs[1:])
    if dirname_str:
        dirname_str += "/"

    basename = path.name
    name, _, longext = basename.partition(".")

    lang: str | None = None
    template: str | None = None
    ext = longext.split(".")[-1]
    for check_lang in [basename, longext, ext]:
        if lang is None:
            lang = LANG.get(check_lang)
        if template is None:
            template = TEMPLATES.get(check_lang)
    if not lang:
        print(f"No known language for '.{ext}' files")
    if not template:
        print(f"No configured template for '.{ext}' files")
    if not lang or not template:
        raise SystemExit(1)

    top = TOPS[lang]
    nsbeg: list[str] = []
    nsend: list[str] = []
    for subns in namespace_value.split("::"):
        nsbeg.append(f"namespace {subns}\n{{")
        nsend.append(f"}}  // namespace {subns}")

    capabbr = re.sub(r"[^A-Z]+", "", name)
    variables = {
        "longext": longext,
        "ext": ext,
        "hext": "hh" if lang != "C" else "h",
        "modeline": f" -*- {lang} -*- ",
        "name": name,
        "namespace": namespace_value,
        "namespace_begin": "\n".join(nsbeg),
        "namespace_end": "\n".join(reversed(nsend)),
        "basename": basename,
        "dirname": dirname_str,
        "capabbr": capabbr,
        "lowabbr": capabbr.lower(),
        "corecel_ns": "",  # or "celeritas::" or someday(?) "corecel::"
        "celeritas_ns": "",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    content = (top + template).format(**variables)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)
        if top.startswith("#!"):
            mode = path.stat().st_mode
            mode |= 0o111
            path.chmod(mode)
    return str(filename)


def get_main_repo() -> Path:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.SubprocessError:
        return Path("..")
    return Path(completed.stdout.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="+", help="file names to generate")
    parser.add_argument(
        "--repodir", type=Path, help="root source directory for file naming"
    )
    parser.add_argument(
        "-o",
        "--open",
        action="store_true",
        help='call "open" on the created files',
    )
    parser.add_argument(
        "--namespace",
        "-n",
        default=None,
        help="C++ namespace to generate",
    )
    args = parser.parse_args(argv)

    repodir = args.repodir or get_main_repo()
    generated: list[str] = []
    for fn in args.filename:
        created = generate(repodir, fn, args.namespace)
        if created:
            generated.append(created)

    if args.open and generated:
        subprocess.run(["open", *generated], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
