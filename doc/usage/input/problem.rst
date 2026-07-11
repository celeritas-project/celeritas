.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_problem:

Problem setup
=============

Virtually all attributes of the problem and its execution are defined through
the ``Problem`` input class, which is set up in part by the user and in part by
external applications as directed by the user.

.. celerstruct:: inp::Problem

The ``OpticalProblem`` input class provides the corresponding configuration for
problems in which only optical photons are transported. It shares the same
overall structure as ``Problem`` but includes options specific to optical
simulations.

.. celerstruct:: inp::OpticalProblem
