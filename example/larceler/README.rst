LArSoft integration
===================

This is a beginning-to-end demonstration of running Celeritas on GPU on ORNL's
ExCL_ cluster using the Fermilab development apptainer image and a
host-provided CUDA.

.. _ExCL: https://docs.excl.ornl.gov

Launch apptainer on milan2
--------------------------

Launch the ``fnal-dev-sl7:latest`` apptainer with CUDA forwarding enabled:

.. literalinclude:: ../../scripts/env/excl.sh
   :language: sh
   :dedent: 2
   :start-after: BEGIN_LARCELER_EXAMPLE_APPTAINER
   :end-before: END_LARCELER_EXAMPLE_APPTAINER

This command is wrapped into the ``apptainer_fermilab`` shell command when
:file:`scripts/env/excl.sh` is sourced.


Build Celeritas
---------------

The :ref:`build_script` should be used to install Celeritas. It uses CUDA
configuration from :file:`scripts/env/milan2.sh` and UPS LArSoft configuration
from :file:`scripts/env/fnal-dev-sl7.sh`.

See :ref:`plugins_larsoft`.


Run LArSoft
-----------

TODO: see `#2318`_

.. _#2318: https://github.com/celeritas-project/celeritas/pull/2318


.. literalinclude:: ../../example/larceler/dune10kt_1x2x6_cpu.fcl
   :language: none
   :start-at: #include


.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0
