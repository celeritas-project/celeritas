#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasUtils
--------------

CMake utility functions for Celeritas.

.. command:: set_required

  Pass the given compiler-dependent warning flags to a library target::

    set_required(<OUTPUT_VAR>
                 <OPTION_VAR>)

  ``OUTPUT_VAR``
    Set variable in the parent scope to "REQUIRED" or empty.

  ``OPTION_VAR``
    Variable to test.

#]=======================================================================]

function(set_required OUTPUT_VAR INPUT_VAR)
  if (INPUT_VAR)
    set(OUTPUT_VAR REQUIRED PARENT_SCOPE)
  else()
    set(OUTPUT_VAR "" PARENT_SCOPE)
  endif()
endfunction()

#-----------------------------------------------------------------------------#
