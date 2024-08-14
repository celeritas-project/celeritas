.. Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_stepping:

Stepping mechanics
==================

The core algorithm in Celeritas is to perform a *loop interchange*
:cite:`allen_automatic_1984` between particle tracks and steps. The classical
(serial) way of simulating an event is to have an outer loop over tracks and an
inner loop over steps, and inside each step are the various actions applied to
a track such as evaluating cross sections, calculating the distance to the
nearest geometry boundary, and undergoing an interaction to produce
secondaries. In Python pseudocode this looks like:

.. code-block:: python

   track_queue = primaries
   while track_queue:
      track = track_queue.pop()
      while track.alive:
         for apply_action in [pre, along, post]:
            apply_action(track)
         track_queue += track.secondaries

There is effectively a data dependency between the track at step *i* and step
*i + 1* that prevents vectorization. The approach Celeritas takes to
"vectorize" the stepping loop on GPU is to have an outer loop over "step
iterations" and an inner loop over "track slots", which are elements in a
fixed-size vector of tracks that may be in flight:

.. code-block:: python

   initializers = primaries
   track_slots = [None] * num_track_slots
   while initializers or any(track_slots):
      fill_track_slots(track_slots, initializers)
      for apply_action in [pre, along, post]:
         for (i, track) in enumerate(track_slots):
            apply_action(track)
            track_queue += track.secondaries
      if not track.alive:
         track_slots[i] = None


The stepping loop in Celeritas is therefore a sorted loop over "actions", each
of which is usually a kernel launch (or an inner loop over tracks if running on
CPU).

Particle states
---------------

.. doxygenenum:: celeritas::TrackStatus
   :no-link:

Action sequence
---------------

.. doxygenenum:: celeritas::ActionOrder
   :no-link:

Execution
---------

.. doxygenclass:: celeritas::Stepper

