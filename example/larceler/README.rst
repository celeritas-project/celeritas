LArSoft workflow example
========================

This example assumes that:

- A suitable Apptainer container has been activated (see :ref:`apptainer_env`),
- LArSoft/DUNESW have been loaded via UPS (see :ref:`ups_mrb`),
- Celeritas has been installed, and
- Celeritas environment variables have been loaded (see :ref:`plugins_larsoft`).

Summary
-------


Overview
--------

The following flow chart shows every step of the workflow,
their respective ``fcl`` job file, along with the generated data type and
their ``ModuleLabel``.

.. figure:: /_static/dot/larsim.*
   :align: center
   :width: 80%

Generating GENIE samples
------------------------


Running LArG4 + IonAndScint
---------------------------


.. _run_lar_optical:

Running optical simulations
---------------------------

.. code::

   $ lar -c dune10kt_1x2x6_cpu.fcl -s genie-output.root -o larg4-output.root


.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0
