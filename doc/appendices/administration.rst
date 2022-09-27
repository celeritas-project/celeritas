.. Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _administration:

**************
Administration
**************

This appendix includes administrative details describing the roles and
responsibilities of participants.


Community standards
===================

Be respectful. Communicate openly. Resolve disagreements peacefully (or even
agree peacefully to postpone their resolution).


.. _roles:

Roles
=====

The roles of the Celeritas code base are related but not identical to the roles
set out in the SciDAC project management plan (PMP). These roles should be
reflected in the different teams and/or access restrictions in GitHub (TODO:
they currently are not).

Code lead
   The code lead (currently `@sethrj`_) is responsible for steering the
   technical implementation of the codebase to meet the long-term project
   goals. This should be the same person as the "core capability lead"
   described in the PMP. One key responsibility of the code lead is to set up
   milestones for releases and execute the release process described below.

Maintainer
   Maintainers should be familiar with most or all parts of the codebase and
   are responsible for merging pull requests and closing issues.

Core team member
   The "core team" are those currently funded to work on Celeritas. Core team
   members are responsible for reviewing pull requests in accordance with the
   :ref:`contributing guidelines <contributing>`. They should regularly
   contribute new code or publish new results using Celeritas.

Contributor
   Anyone can submit a pull request that conforms to the contribution
   guidelines! Contributors are recognized by including their handle and pull
   requests in release notes.

.. _@sethrj: https://github.com/sethrj


Release process
===============

Celeritas uses `Semantic Versioning`_ to enumerate releases. During its initial
development phase, ``0.x.0`` is a major release and ``0.x.z`` is a patch
release. When Celeritas is declared stable, ``x.0.0`` is a major release,
``x.y.0`` is a minor release, and ``x.y.z`` is a patch release.

Major and minor releases (including 0.x.0 development releases) must have a
milestone in the git issue tracker with a list of issues that can be assigned.
Only major releases can deprecate or remove features and change
:ref:`public-facing APIs <api>`. Both major and minor releases should include
notable improvements to the code.

Patch releases can be created at any time but should typically include at least
one critical bug fix or several substantial fixes. Patch releases should focus
almost exclusively on fixes and should generally not include new features or
other major code changes.

.. _Semantic Versioning: https://semver.org


Releases can be created from the master branch (major, minor, patch) or a
"backport" branch (minor, patch). The following process must be followed (and
may need iteration to converge) for each release.

- Ensure all CI jobs passed for the release in question. This is automatic for
  releases from master (since every pull request must pass) but should be
  checked manually for backports.
- Run regression tests on Summit, Crusher, Wildstyle to check for major
  performance regressions or newly failing tests. Postpone the release
  temporarily if major new bugs are detected. If minor updates are needed to
  fix the build or tests on a particular machine, include those as part of the
  "pre-release" pull request that includes new documentation.
- [TODO: define high-level validation tests and a test matrix correlating
  capability areas (code files/directories changed) to test names.] Rerun
  and check all validation tests that might be affected by changes since the
  previous release.
- Update documentation with release notes from all pull requests newly included
  in the release. Follow the format for previous releases: add a summary of
  highlights, and enumerate the pull requests (with PR numbers and
  authorship attribution) separated by features and bug requests. [TODO:
  automate this using pull request tags and the GitHub API]
- Ensure the code documentation builds, preferably without warnings, on a
  configuration that has Sphinx, Doxygen, and Breathe active.
- Submit a pull request with the newly added documentation and any
  release-related tweaks, and wait until it's reviewed and merged.
- If releasing a backported version branch, cherry-pick this documentation
  commit into the backport branch.
- Use the GitHub interface to create a new release with the documentation
  update that was just added.

After committing the release tag:

- Save the ``tar.gz`` and attach to the release, because the hash changes if the
  git "describe" function returns a different result for the release tag's hash
  (e.g., if "master" is a branch that points to it).
- Pull locally (make sure to use the ``--tags`` option) and build PDF user
  documentation for the release. Ensure breathe is activated (so the API is
  listed) and that the version is embedded correctly.  [TODO: We should add a
  documentation pipeline that builds and uploads to GitHub pages.]
- Update the Spack recipe for Celeritas with the new version and sha256 value
  (either manually or using ``spack checksum``) and submit a pull request to
  the Spack project.
