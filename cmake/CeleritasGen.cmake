#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasGen
------------

Automatically generate headers and source files for dual CPU/GPU kernel launches.

.. command:: celeritas_gen

  Generate from the given class and function name::

    celeritas_gen(<var> <script> <subdir> <basename> [...])

      ``var``
        Variable name to append created source file names in the parent scope.

      ``script``
        Python script name inside ``cmake/CeleritasUtils/``

      ``subdir``
        Local source subdirectory in which to generate files

      ``basename``
      Name without exension.

#]=======================================================================]

include_guard(GLOBAL)

#-----------------------------------------------------------------------------#

function(celeritas_gen var script basename)
  set(_scriptdir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CeleritasGen")
  set(script "${_scriptdir}/${script}")
  set(_path_noext "${CMAKE_CURRENT_SOURCE_DIR}/${basename}")

  if(Python_FOUND)
    # Regenerate files on the fly
    add_custom_command(
      COMMAND "$<TARGET_FILE:Python::Interpreter>"
        "${script}"
        --basename ${basename} ${ARGN}
      OUTPUT "${_path_noext}.cc" "${_path_noext}.cu" "${_path_noext}.hh"
      DEPENDS
        "${script}"
        "${_scriptdir}/launchbounds.py"
        "${_scriptdir}/launch-bounds.json"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()

  set(_sources ${${var}} "${_path_noext}.cc")
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    list(APPEND _sources "${_path_noext}.cu")
  endif()
  if(CELERITAS_USE_HIP)
    set_source_files_properties("${_path_noext}.cu"
      PROPERTIES LANGUAGE HIP
    )
  endif()
  set(${var} "${_sources}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
