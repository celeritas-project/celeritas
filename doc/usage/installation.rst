.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. **NOTE**: this file is referenced by README.md:
.. if changing the former, update the latter!!

.. highlight:: console

.. _build_ups:

UPS for LArSoft
---------------

Building Celeritas for LArSoft or DUNE (see :ref:`plugins_larsoft`) is
straightforward once the Fermilab-developed UPS_ build environment has been set up.

.. _UPS: https://cdcvs.fnal.gov/redmine/projects/ups/wiki/Getting_Started_Using_UPS

.. note:: UPS and these images are in the process of being replaced with a
   Spack toolchain. If you are using a Spack-based distribution of
   larsoft/dunesw already, you should be able to install Celeritas with the
   standard instructions above.

.. _apptainer_env:

Apptainer
^^^^^^^^^

UPS-based builds always happen within a containerized system. These
instructions demonstrate container execution for two use cases: using CUDA on the ExCL milan2_ system, and without CUDA on Fermilab's ``scisoftbuild01`` machine.

To enable CUDA, launch the ``fnal-dev-sl7:latest`` Apptainer_ image, stored on
CVMFS_, with CUDA forwarding enabled (and the CUDA directory forwarded via
``-B``):

.. literalinclude:: ../../scripts/env/excl.sh
   :language: sh
   :dedent: 2
   :start-after: BEGIN_DOC_APPTAINER
   :end-before: END_DOC_APPTAINER

This command is wrapped into the ``apptainer-fnal`` shell command when
:file:`scripts/env/excl.sh` is sourced.

On Fermilab machines, most of which require Kerberos authentication and do
*not* have CUDA support, omit the ``--nv`` flag and forward the hosts files.

.. literalinclude:: ../../scripts/env/scisoftbuild01.sh
   :language: sh
   :dedent: 2
   :start-after: BEGIN_DOC_APPTAINER
   :end-before: END_DOC_APPTAINER

.. _milan2: https://docs.excl.ornl.gov/system-overview/milan
.. _CVMFS: https://cvmfs.readthedocs.io/en/stable/
.. _Apptainer: https://apptainer.org/docs/user/main/


.. important:: Because the ``fnal-dev-sl7`` uses a *very* old operating system,
   the default LArG4 installation will likely fail to load when enabling CUDA
   with the ``--nv`` flag, which forwards a number of host libraries to the
   container. If this happens, you will see an error:

   .. code::
     :language: none
     Unable to load requested library .../liblarg4_Services_LArG4Detector_service.so
     /lib64/libc.so.6: version 'GLIBC_2.38' not found (required by /.singularity.d/libs/libGLX.so.0)

   This is due to Geant4's visualization functionality (which uses OpenGL).
   It can be fixed by commenting out the lines in
   :file:`{/etc}/apptainer/nvliblist.conf` that start with libGL and libgl.

.. _ups_mrb:

UPS and MRB
^^^^^^^^^^^

To set up Celeritas dependencies for minimal LArSoft development:

.. literalinclude:: ../../scripts/env/fnal-dev-sl7.sh
   :language: sh
   :dedent: 2
   :start-after: BEGIN_DOC_UPS
   :end-before: END_DOC_UPS

The ``-q`` qualifiers_ denote the compiler version and flags.
These dependencies are loaded automatically when using the ``build.sh`` script
inside the Apptainer image.

.. _qualifiers: https://cdcvs.fnal.gov/redmine/projects/cet-is-public/wiki/AboutQualifiers

.. tip::

   Use the command :samp:`ups list -aK+ {package}` to list available packages.

Alternatively, for integration into DUNE_ development environment:

.. code::

   $ source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
   Setting up larsoft UPS area... /cvmfs/larsoft.opensciencegrid.org
   Setting up DUNE UPS area... /cvmfs/dune.opensciencegrid.org/products/dune/
   $ setup dunesw v10_20_00d00 -q e26:prof

If using MRB_ with at least one repository (i.e. you called ``mrb g ...``),
``cmake`` will be available in your ``$PATH``.

.. _DUNE: https://github.com/DUNE/dunesw/releases
.. _MRB: https://cdcvs.fnal.gov/redmine/projects/mrb/wiki/MrbUserGuide

Installing Celeritas
^^^^^^^^^^^^^^^^^^^^

Celeritas does not currently have a UPS package.
Instead, build and install it like any other CMake package, using the build
script, presets, or manually:

.. code::

   $ git clone https://github.com/celeritas-project/celeritas.git
   $ cd celeritas
   $ cmake --preset=larsoft .
   $ cmake --preset=larsoft --install .

On some machines such as Perlmutter, which has Nvidia's HPC SDK installed, you
may need additional setup inside a container to configure Celeritas with CUDA:

.. code:: sh

   function export-native-cuda() {
       HPCSDK_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5"
       export CUDA_HOME="$HPCSDK_DIR/cuda/12.9"
       export PATH="$CUDA_HOME/bin":$PATH
       export CUDACXX="$CUDA_HOME/bin/nvcc"
       export CUDAARCHS=80 # For Nvidia A100

       export CPATH="$HPCSDK_DIR/math_libs/12.9/include:$CPATH"
       export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$CUDA_HOME/extras/Debugger/lib64:$HPCSDK_DIR/math_libs/12.9/lib64:$LD_LIBRARY_PATH"
   }
