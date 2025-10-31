.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_stepping:

Stepping mechanics
==================

The core algorithm in Celeritas is to perform a *loop interchange*
:cite:`allen-loop-1984` between particle tracks and steps. The classical
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


The stepping loop in Celeritas is therefore a sorted loop over "step actions",
each of which is usually a kernel launch (or an inner loop over tracks if
running on CPU).

Actions
-------

Actions can operate on shared parameters and thread-local state collections.
All actions inherit from a :cpp:class:`celeritas::ActionInterface` abstract
base class, and the hierarchy of actions allows multiple inheritance so that a
single "action" class can, for example, allocate states at the beginning of the
run and execute once per step.

There are currently two different actions that act as extension points to the
stepping loop: :cpp:class:`celeritas::BeginRunActionInterface` is called once
per event (or set of simultaneously initialized events), and
:cpp:class:`celeritas::StepActionInterface`
is called once per step, ordered using :cpp:enum:`celeritas::StepActionOrder`.

.. doxygenclass:: celeritas::ActionInterface
.. doxygenclass:: celeritas::BeginRunActionInterface
.. doxygenclass:: celeritas::StepActionInterface

.. doxygenenum:: celeritas::StepActionOrder
   :no-link:


Initialization and execution
----------------------------

- The front end constructs the :ref:`api_physics_def` classes and allows user
  actions and :ref:`api_auxiliary_data` to be set up
- "Core params", which reference these classes, are constructed; in the
  process, certain required implementation actions (e.g., managing primaries
  and secondaries, initializing tracks, crossing boundaries) are added to the
  action
- Additional user actions and data can be added
- The "core state" is created on each CPU thread (or task), simultaneously
  constructing a vector of auxiliary state data
- The :cpp:class:`celeritas::Stepper` constructs a final ordered runtime vector
  of actions
- The Stepper immediately calls the "begin run" actions
- Each step calls all the "step" actions

.. doxygenenum:: celeritas::TrackStatus
   :no-link:

.. doxygenclass:: celeritas::Stepper

Track sort order
----------------

For performance reasons such as reducing divergence and improving memory access
patterns, it is desirable to map similar tracks into similar threads. There
will be an upcoming paper describing and analyzing these options in more
detail.

.. doxygenenum:: celeritas::TrackOrder
   :no-link:
