LArSoft workflow example
========================

This example assumes that:

- A suitable Apptainer container has been activated (see :ref:`apptainer_env`),
- DUNESW has been loaded via UPS (see :ref:`ups_mrb`),
- Celeritas has been installed against a compatible LArSoft version, and
- Celeritas environment variables have been loaded (see :ref:`plugins_larsoft`).

Summary
-------

This sequence of commands generate analysis files for Celeritas and the
LArSim fast simulation using a single GENIE-generated event.

.. code:: sh

   # Download and patch the geometry file for Celeritas execution
   curl https://raw.githubusercontent.com/nuRiceLab/laropticks/refs/heads/main/laropticks/GDML/dune10kt_v6_refactored_1x2x6.gdml > dune10kt_v6_refactored_1x2x6.gdml
   patch -p0 -o dune10kt_v6_refactored_1x2x6_zwires_celeritas.gdml < dune10kt_v6_refactored_1x2x6_zwires_celeritas.patch
   # Run GENIE
   lar -c prodgenie_nu_dune10kt_1x2x6.fcl -n 1 -o genie-output.root
   # Run LArG4 + IonAndScint
   lar -c larg4_dune10kt_1x2x6.fcl -s genie-output.root -o larg4-output.root
   # Run FastSim and Celeritas optical simulations
   lar -c opticalsim_dune10kt_1x2x6.fcl -s larg4-output.root -o fastsim-output.root
   lar -c opticalsim_celeritas_dune10kt_1x2x6.fcl -s larg4-output.root -o celeritas-output.root
   # Generate analysis files
   lar -c pdsimana_job.fcl -s fastsim-output.root
   lar -c pdsimana_job.fcl -s celeritas-output.root

Overview
--------

The set of FHiCL job files in this example can

- produce neutrino events with GENIE,
- simulate the energy deposition in LAr with Geant4,
- propagate optical photons via fast simulation and Celeritas, and
- generate analysis files.

The flow chart :numref:`fig-larsim` shows every step of the workflow,
their respective ``fcl`` job file, along with the generated data type and
their ``ModuleLabel``.

.. _fig-larsim:

.. figure:: /_static/dot/larsim.*
   :align: center
   :width: 80%

   LArSoft workflow and FHiCL file descriptions.

The next sections describe each component and their invocation, as well
as a description of setting up a ``dunesw`` environment with native
(non-spack) CUDA.

Generating GENIE samples
------------------------

- Default
  `prodgenie_nu_dune10kt.fcl <https://internal.dunescience.org/doxygen/prodgenie__nu__dune10kt__1x2x6_8fcl_source.html>`__,
  which is in your ``PATH`` through ``dunesw``.

.. code:: console

   $ lar -c prodgenie_nu_dune10kt_1x2x6.fcl -n 10 -o genie-output.root

- If ``-n`` is not used, the default number of events is set to 10
  (defined upstream in
  `prodgenie_common_dunefd.fcl <https://internal.dunescience.org/doxygen/prodgenie__common__dunefd_8fcl_source.html>`__).
- The GENIE input configuration is also upstream, at
  `genie_dune.fcl <https://internal.dunescience.org/doxygen/genie__dune_8fcl_source.html>`__
  (see ``Configurations for 1x2x6 geometry``).

Running LArG4 + IonAndScint
===========================

- Use local ``larg4_dune10k_1x2x6.fcl`` file.
- To loop over a subset of events, replace ``-s`` by
  ``-n [num_events]``.
- The GDML input geometry in ``LArG4`` should **not** tag Arapucas as
  ``SensDet``.

.. code:: sh

   $ lar -c larg4_dune10kt_1x2x6.fcl -s genie-output.root -o larg4-output.root

.. _run_lar_optical:

Running optical simulations
---------------------------

- Use local ``opticalsim*.fcl`` files.
- To loop over a subset of events, replace ``-s`` by
  ``-n [num_events]``.

Fast simulation
^^^^^^^^^^^^^^^

.. code:: console

   $ lar -c dune10k_optical_1x2x6.fcl -s larg4-output.root -o fastsim-output.root

Celeritas
^^^^^^^^^

.. code:: console

   $ lar -c dune10k_optical_celeritas_1x2x6.fcl -s larg4-output.root -o celeritas-output.root

Celeritas geometry requires correct optical material information and correct
``SensDet`` data assigned to the Arapucas. The geometry has also been modified to
remove the U and V wires, which cause significant slowdowns in the BIH geometry
navigation. To patch the geometry with the changes needed by Celeritas, apply
the local patch file to the ``dune10kt_v6_refactored_1x2x6.gdml``, available at
`nuRiceLab <https://github.com/nuRiceLab/laropticks/blob/main/laropticks/GDML/dune10kt_v6_refactored_1x2x6.gdml>`__.

To patch the file:

.. code:: console

   $ patch -p0 -o dune10kt_v6_refactored_1x2x6_zwires_celeritas.gdml < dune10kt_v6_refactored_1x2x6_zwires_celeritas.patch


Generating analysis files from the optical simulation
-----------------------------------------------------

.. code:: cnosole

   $ lar -c pdsimana_job.fcl -s fastsim-output.root
   $ lar -c pdsimana_job.fcl -s celeritas-output.root

- Use local ``pdsimana_job.fcl`` file (in ``src/larceler``).
- To loop over a subset of events, replace ``-s`` by
  ``-n [num_events]``.
- Optional ``-T filename.root``: Overrides the default analyzer output file
  naming scheme, updating the ``services.TFileServices.fileName`` field. This is
  equivalent to passing the full
  ``--services.TFileService.fileName=my-output.root`` path directly to ``lar``.
- As noted in the ``pdsimana.fcl`` documentation, the
  ``ModuleLabel: "OpticalSim"`` in ``pdsimana.fcl`` is correct if
  optical simulation is generated with the local ``*optical*.fcl``
  files. If fast simulation is generated from a default LArSoft ``fcl``
  job, that will likely be ``PDFastSim``.

Note on ``ModuleLabel``
^^^^^^^^^^^^^^^^^^^^^^^

- ``art`` stores the module labels as part of the object naming
  convention inside an ``art::Event`` object. If unsure about what label
  to use while trying to load an object type in ``PDSimAna``, you can
  view it directly on ROOT. E.g., for ``SimEnergyDeposits`` objects, the
  ``IonAndScint`` label is shown as part of the branch name:

.. code:: none

   $ root art-file.root
   root[1] Events->Print()
   .
   .
   .
   *............................................................................*
   *Br   21 :sim::SimEnergyDeposits_IonAndScint__geantionandscint.obj :         *
   *         | vector<sim::SimEnergyDeposit>                                    *
   *Entries :       10 : Total  Size=   61060255 bytes  File Size  =   30434130 *
   *Baskets :       10 : Basket Size=      16384 bytes  Compression=   2.01     *
   *............................................................................*
   .
   .
   .

.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0
