.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _claude_plugin:

AI coding assistant plugin
==========================

Celeritas provides a plugin for the `Claude Code`_ AI coding assistant. It
helps users who *integrate* Celeritas, rather than develop it: linking
Celeritas from CMake, adding offload to a Geant4 application, and diagnosing
Celeritas runs. The plugin does not need a Celeritas source checkout.

.. _Claude Code: https://code.claude.com

Installation
------------

In Claude Code, add the Celeritas repository as a plugin marketplace and
install the plugin:

.. code-block:: none

   /plugin marketplace add celeritas-project/celeritas
   /plugin install celeritas@celeritas

The marketplace follows the default branch unless a tag is given. To match the
skills to an installed Celeritas release, pin the marketplace to that
release's tag, e.g. ``/plugin marketplace add
celeritas-project/celeritas@vX.Y.Z``. Only releases made after the plugin was
added contain it. Claude Code does not automatically update third-party
marketplaces: run ``/plugin marketplace update celeritas`` to pick up changes.

Skills
------

Claude loads each skill automatically when a request matches it, and each can
also be invoked directly:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Skill
     - Purpose
   * - ``/celeritas:doctor``
     - Detect the installed Celeritas version and build options (GPU,
       geometry, MPI, debug), the Geant4 version and data, and GPU
       visibility, and flag inconsistencies such as a GPU architecture
       mismatch.
   * - ``/celeritas:cmake-link``
     - Write or fix downstream CMake code: ``find_package``,
       ``celeritas_target_link_libraries``, CUDA relocatable device code, and
       plugin libraries (see :ref:`library`).
   * - ``/celeritas:geant4-offload``
     - Add or adjust offloading in a Geant4 application: integration choice
       by Geant4 version, :cpp:class:`celeritas::SetupOptions`, sensitive
       detectors, magnetic fields, UI macros, and validation against pure
       Geant4.
   * - ``/celeritas:troubleshoot``
     - Match Celeritas error messages to causes and fixes, and summarize the
       ``celeritas.out.json`` diagnostic output.

Design and maintenance
----------------------

The plugin lives in :file:`contrib/claude-plugin`, and the repository root
holds the marketplace definition :file:`.claude-plugin/marketplace.json`. Its
skills describe workflows and common mistakes. For API details they direct
the assistant to the user's *installed* Celeritas headers and to this manual,
so the advice tracks the installed version.

The plugin manifest deliberately has no ``version`` field. Claude Code then
versions the plugin by git commit, so no release step is needed to publish
updates.

The script :file:`scripts/ci/check-claude-plugin.py` runs as a pre-commit
hook. It fails if the plugin refers to a repository file, environment
variable, ``SetupOptions`` member, ``/celer/`` UI command, or ``celeritas::``
class that no longer exists. When changing any of these, update the plugin in
the same pull request.
