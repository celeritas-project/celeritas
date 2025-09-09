.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _em_photon_interaction:

Photon interactions
===================

Scattering
----------

.. doxygenclass:: celeritas::KleinNishinaInteractor
.. doxygenclass:: celeritas::RayleighInteractor


Photoelectric effect
--------------------

.. doxygenclass:: celeritas::LivermorePEInteractor
.. doxygenclass:: celeritas::AtomicRelaxation

Livermore photoelectric cross sections are calculated
on the fly.

.. doxygenclass:: celeritas::LivermorePEMicroXsCalculator
