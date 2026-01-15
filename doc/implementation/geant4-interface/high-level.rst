.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_accel_high_level:

High level interfaces
=====================

The high-level integration classes are the easiest way to add Celeritas to a
Geant4 application. Under the hood, it contains a singleton class instance that
sets up the UI commands (see :cpp:class:`celeritas::SetupOptionsMessenger`),
MPI (if configured), and Celeritas logging.

.. doxygenclass:: celeritas::IntegrationBase

Tracking manager
----------------

Using Celeritas to "offload" all electrons, photons, and gammas from Geant4 can
be done using the new-ish Geant4 interface :cpp:class:`G4VTrackingManager`
implemented by :cpp:class:`celeritas::TrackingManager`. To set up the tracking
manager correctly, we recommend using this helper class:

.. doxygenclass:: celeritas::TrackingManagerConstructor

The high-level :cpp:class:`celeritas::TrackingManagerIntegration` class should be used in
addition to the tracking manager constructor to set up and tear down Celeritas.
See :ref:`example_template` for a template of adding to a user application.

.. doxygenclass:: celeritas::TrackingManagerIntegration
   :members:

Fast simulation
---------------

It is currently *not* recommended to offload tracks on a per-region basis, since
tracks exiting that region remain in Celeritas and on GPU.

.. doxygenclass:: celeritas::FastSimulationModel
.. doxygenclass:: celeritas::FastSimulationIntegration
   :members:

User action
-----------

For compatibility with older versions of Geant4, you may use the following
class to integrate Celeritas by manually intercepting tracks with a
``UserTrackingAction``.

.. doxygenclass:: celeritas::UserActionIntegration
   :members:
