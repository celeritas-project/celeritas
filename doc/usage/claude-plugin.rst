.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _claude_plugin:

AI coding assistant plugin
==========================

Celeritas provides a plugin for the `Claude Code`_ AI coding assistant. It
helps users who *integrate* Celeritas, rather than develop it: installing and
linking Celeritas, adding offload to Geant4 applications and experiment
frameworks, running the standalone applications, tuning performance, and
diagnosing runs. The plugin does not need a Celeritas source checkout.

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
release's tag, e.g. ``/plugin marketplace add celeritas-project/celeritas#vX.Y.Z``.
Claude Code does not automatically update third-party marketplaces:
run ``/plugin marketplace update celeritas`` to pick up changes.

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
   * - ``/celeritas:install``
     - Choose an installation route and configuration: Spack variants, CMake
       options and presets, and geometry/GPU/Geant4 compatibility (see
       :ref:`infrastructure`).
   * - ``/celeritas:cmake-link``
     - Write or fix downstream CMake code: ``find_package``,
       ``celeritas_target_link_libraries``, CUDA relocatable device code, and
       plugin libraries (see :ref:`library`).
   * - ``/celeritas:geant4-offload``
     - Add or adjust offloading in a Geant4 application: integration choice
       by Geant4 version, :cpp:class:`celeritas::SetupOptions`, sensitive
       detectors, magnetic fields, UI macros, and validation against pure
       Geant4.
   * - ``/celeritas:framework-integration``
     - Integrate with experiment frameworks: the DD4hep/DDG4 ``CelerPhysics``
       plugin, the LArSoft ``PDFullSimCeler`` module (see
       :ref:`plugins_larsoft`), or custom and task-based frameworks such as
       Gaudi, CMSSW, and Athena.
   * - ``/celeritas:review-integration``
     - Review existing integration code for problems, using the
       ``celeritas:integration-reviewer`` agent described below.
   * - ``/celeritas:new-offload-app``
     - Start a new Geant4 application from the offload template (see
       :ref:`example_template`) at the tag matching the installed version.
   * - ``/celeritas:standalone-input``
     - Write or update input for ``celer-sim``, ``celer-optical``,
       ``celer-g4``, ``celer-geo``, and ``celer-export-geant`` (see
       :ref:`json-input`).
   * - ``/celeritas:library-api``
     - Use Celeritas from C++ without Geant4 offload: programmatic standalone
       transport, geometry-only use, step data collection, and custom GPU
       step actions (see :ref:`api`).
   * - ``/celeritas:performance-tuning``
     - Estimate the achievable speedup, size track capacities, read timers,
       and profile (see :ref:`profiling`).
   * - ``/celeritas:troubleshoot``
     - Match Celeritas error messages to causes and fixes, and summarize the
       ``celeritas.out.json`` diagnostic output.

The ``celeritas:integration-reviewer`` agent audits integration code (setup
order, threading, options, sensitive detectors, MC truth, CMake) without
modifying it, and returns ranked findings.

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
hook. It fails if the plugin refers to a repository file or header,
environment variable, ``SetupOptions`` member, ``/celer/`` UI command, or
``celeritas::`` class that no longer exists. When changing any of these,
update the plugin in the same pull request.

The plugin also has an evaluation suite in
:file:`contrib/claude-plugin/evals`, run with ``claude plugin eval``. Each case
gives Claude a realistic task, such as adding offload to the Geant4 example,
fixing CUDA linking, or reviewing a broken integration. It then grades the
result, both with and without the plugin. Cases are tagged ``smoke`` (cheap,
no setup), ``fixture`` (copies files from :file:`example/geant4`), and
``build`` (compiles and runs against a Geant4 and Celeritas install outside
the home directory).
Running the suite starts real Claude sessions billed to the user's
credentials, so maintainers run it locally after substantive skill changes and
before releases. The plugin's :file:`README.md` lists the commands.
