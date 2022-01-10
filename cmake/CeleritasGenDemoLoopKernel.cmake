#----------------------------------*-CMake-*----------------------------------#
# Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasGenDemoLoopKernel
--------------------------

Automatically generate headers and source files for dual CPU/GPU launching of
demo loop kernels.

.. command:: celeritas_gen_demo_loop_kernel

  Generate from the given class and function name::

    celeritas_gen_demo_loop_kernel(<var> <class> <func>)

      ``var``
        Variable name to append created source file names in the parent scope.

      ``class``
        EM physics class name (e.g. "BetheHeitler")

      ``func``
        Lower case name used in the kernel launch command.

      ``threads``
        Number of threads.

#]=======================================================================]

set(CELERITAS_GEN_DEMO_LOOP_KERNEL
    "${PROJECT_SOURCE_DIR}/scripts/dev/gen-demo-loop-kernel.py"
    CACHE INTERNAL "Path to gen-demo-loop-kernel.py")

function(celeritas_gen_demo_loop_kernel var class func threads)
  set(_srcdir "${PROJECT_SOURCE_DIR}/app")
  set(_subdir "demo-loop/generated")
  set(_basename "${_subdir}/${class}Kernel")

  if(PYTHON_FOUND)
    # Regenerate files on the fly
    add_custom_command(
      COMMAND "${Python_EXECUTABLE}"
        "${CELERITAS_GEN_DEMO_LOOP_KERNEL}"
        --class ${class} --func ${func} --threads ${threads}
      OUTPUT "${_srcdir}/${_basename}.cc" "${_srcdir}/${_basename}.cu"
      DEPENDS "${CELERITAS_GEN_DEMO_LOOP_KERNEL}"
      WORKING_DIRECTORY
        "${PROJECT_SOURCE_DIR}/app/${_subdir}"
    )
  endif()

  set(_sources ${${var}} "${_basename}.cc")
  if(CELERITAS_USE_CUDA)
    list(APPEND _sources "${_basename}.cu")
  endif()
  set(${var} "${_sources}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
