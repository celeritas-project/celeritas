*****************
Quick start guide
*****************

.. only:: github

   You may be viewing this document through GitHub's native viewer. if so, the
   links below may not work; please visit this page on the documentation web
   site:

   https://celeritas-project.github.io/celeritas/user/index.html#quick-start-guide


Common use cases to get started with Celeritas:

1. Loading as a LArSoft plugin:

   - Build with UPS (:ref:`build_ups`) or the Fermilab spack extension (TODO).
   - Start your analysis container (:ref:`apptainer_env`).
   - Load the Celeritas plugins (:ref:`plugins_larsoft`) with
     ``eval $($CELER_DIR/bin/larceler-env)``.
   - Run LArSoft (:ref:`run_lar_optical`).

2. Integrating as a library into an existing Geant4 application:

   - Configure, build, and install Celeritas with the Geant4 directory in your
     ``CMAKE_PREFIX_PATH``.
   - In your application's CMake, add ``find_package(Celeritas)`` and link
     against ``Celeritas::G4`` (see :ref:`library` and :ref:`example_template`).
   - See :ref:`api_g4_interface` for configuring Celeritas in your app.

3. Executing a Celeritas testing app for profiling on HPC:

   - Install Celeritas with an appropriate :ref:`configuration` or using the
     :ref:`build_script`.
   - Set up a test problem input (se :ref:`example_celer_sim`).
   - Run through a profiling front end and analyze the results
     (:ref:`profiling`).

4. Development:

   - Install dependencies with Spack (see :ref:`spack_deps`).
   - Use the developer :ref:`build_script` to install Celeritas.

.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0
