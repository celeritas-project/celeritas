#----------------------------------*-CMake-*----------------------------------#
# Copyright 2021 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasGenInteractor
----------------------

Automatically generate headers and source files for dual CPU/GPU launching of
interactor kernels.

.. command:: celeritas_gen_interactor

  Generate from the given class and function name::

    celeritas_gen_interactor(<var> <class> <func>)

      ``var``
        Variable name to append created source file names in the parent scope.

      ``class``
        EM physics class name (e.g. "BetheHeitler")

      ``func``
        Lower case name used in the kernel launch command.

#]=======================================================================]

set(CELERITAS_GEN_INTERACTOR
    "${PROJECT_SOURCE_DIR}/scripts/dev/gen-interactor.py"
    CACHE INTERNAL "Path to gen-interactor.py")

function(celeritas_gen_interactor var class func)
  set(_srcdir "${PROJECT_SOURCE_DIR}/src")
  set(_subdir "physics/em/generated")
  set(_basename "${_subdir}/${class}Interact")

  if(PYTHON_FOUND)
    # Regenerate files on the fly
    add_custom_command(
      COMMAND "${Python_EXECUTABLE}"
        "${CELERITAS_GEN_INTERACTOR}" --class ${class} --func ${func}
      OUTPUT "${_srcdir}/${_basename}.cc" "${_srcdir}/${_basename}.cu"
      DEPENDS "${CELERITAS_GEN_INTERACTOR}"
      WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/src/${_subdir}"
    )
  endif()

  set(_sources ${${var}} "${_basename}.cc")
  if(CELERITAS_USE_CUDA)
    list(APPEND _sources "${_basename}.cu")
  endif()
  set(${var} "${_sources}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
