#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasGenInteractor
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

set(CELERITAS_GEN_INTERACTOR
    "${PROJECT_SOURCE_DIR}/scripts/dev/gen-interactor.py"
    CACHE INTERNAL "Path to gen-interactor.py")

function(celeritas_gen_interactor class func)
  message(STATUS "Generating interactor: ${class}")
  execute_process(
    COMMAND "${Python_EXECUTABLE}"
      "${CELERITAS_GEN_INTERACTOR}" --class ${class} --func ${func}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/src/physics/em/generated")

  # Add generated sources to parent scope variables (note CMake weirdness of
  # having to create a local variable first)
  set(GEN_SOURCES_CC ${GEN_SOURCES_CC} "physics/em/generated/${class}Interact.cc")
  set(GEN_SOURCES_CC ${GEN_SOURCES_CC} PARENT_SCOPE)
  set(GEN_SOURCES_CU ${GEN_SOURCES_CU} "physics/em/generated/${class}Interact.cu")
  set(GEN_SOURCES_CU ${GEN_SOURCES_CU} PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
