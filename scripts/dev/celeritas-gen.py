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

namespace celeritas {{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
  {name} ...;
   \endcode
 */
class {name} {{
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
}}  // namespace celeritas

#include "{name}.i.{hext}"

#endif // {header_guard}
'''

INLINE_FILE = '''\

namespace celeritas {{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
{name}::{name}() {{
}}

//---------------------------------------------------------------------------//
}}  // namespace celeritas
'''

CODE_FILE = '''\
#include "{name}.{hext}"

namespace celeritas {{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
}}  // namespace celeritas
'''

CMAKE_TOP = '''\
#{modeline:-^77s}#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
'''

CMAKELISTS_FILE = '''\
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
'''


CMAKE_FILE = '''\
#[=======================================================================[.rst:

{basename}
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
    'h': HEADER_FILE,
    'i.h': INLINE_FILE,
    'cc': CODE_FILE,
    'cu': CODE_FILE,
    'cuh': HEADER_FILE,
    'k.cuh': INLINE_FILE,
    'i.cuh': INLINE_FILE,
    't.cuh': INLINE_FILE,
    'cmake': CMAKE_FILE,
    'CMakeLists.txt': CMAKELISTS_FILE,
}

LANG = {
    'h': "C++",
    'cc': "C++",
    'cu': "CUDA",
    'cuh': "CUDA",
    'cmake': "CMake",
}

TOPS = {
    'C++': CLIKE_TOP,
    'CUDA': CLIKE_TOP,
    'CMake': CMAKE_TOP,
}

HEXT = {
    'C++': "h",
    'CUDA': "cuh",
}

def generate(root, filename):
    if os.path.exists(filename):
        print("Skipping existing file " + filename)
        return
    relpath = os.path.relpath(filename, start=root)
    (basename, _, longext) = filename.partition('.')

    template = TEMPLATES.get(filename, None)
    if template is None:
        template = TEMPLATES.get(longext, None)
    if template is None:
        print("Invalid extension ." + longext)
        sys.exit(1)

    ext = longext.split('.')[-1]
    for check_lang in [filename, longext, ext]:
        try:
            lang = LANG[check_lang]
        except KeyError:
            continue
        else:
            break
    else:
        raise KeyError("Missing extension in LANG")

    top = TOPS[lang]

    variables = {
        'longext': longext,
        'ext': ext,
        'hext': HEXT.get(lang, ext),
        'modeline': "-*-{}-*-".format(lang),
        'name': re.sub(r'\..*', '', os.path.basename(filename)),
        'header_guard': re.sub(r'\W', '_', relpath),
        'filename': filename,
        'basename': basename,
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
