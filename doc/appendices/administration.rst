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

Be respectful. Communicate openly. Resolve disagreements peacefully.


Resolution process
------------------

If a disagreement about Celeritas can't be immediately resolved, it's OK to
postpone resolution (maybe you need more data, more time to think, or it's not
important enough at the end of the day). In the case that a resolution is
needed quickly (e.g., if it's a blocking issue), it should be discussed among
the :ref:`core team <roles>` until an agreement is reached. If the issue is
still contentious, the leadership team should make an executive decision,
giving weight to the appropriate capability area lead if possible.


.. _roles:

Roles
=====

The roles of the Celeritas code base are related to the roles
set out in the SciDAC project management plan (PMP). These roles should be
reflected in the different teams and/or access restrictions in GitHub.

Leadership team member
   The "leadership team" comprises the PIs and capability leads in the PMP who
   work directly on the Celeritas code. They are responsible for
   the long-term and big-picture project goals, and they ensure that project
   milestones are met.

Code lead
   The code lead is the "core capability lead" described in the PMP and
   is responsible for steering the technical implementation of the codebase to
   meet the long-term project goals. One key responsibility of the code lead is
   to set up milestones for releases and execute the release process described
   below.

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
   requests in release notes but do not otherwise have a formal role.


Role change process
-------------------

Adding or removing a member of the "core team" must be done by consensus of the
leadership team (or if the core team member wants to remove themself). Adding
maintainers can be done at the whim of the code lead. The `team list`_ on
GitHub is the official record of roles.

.. _team list: https://github.com/orgs/celeritas-project/teams

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


Releases can be created from the main branch (major, minor, patch) or a
"backport" branch (minor, patch). The following process must be followed (and
may need iteration to converge) for each release.

- Ensure all CI jobs passed for the release in question. This is automatic for
  releases from the main branch (since every pull request must pass) but should
  be checked manually for backports.
- Run regression tests on Summit (for performance testing), Crusher (for HIP
  testing), and an additional machine with debug assertions enabled (e.g.
  Wildstyle). Postpone the release
  temporarily if major new bugs or performance regressions are detected. If
  minor updates are needed to fix the build or tests on a particular machine,
  include those as part of the "pre-release" pull request that includes new
  documentation.
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
  (e.g., if a collaborative branch on the main repository points to that commit).
- Pull locally (make sure to use the ``--tags`` option) and build PDF user
  documentation for the release. Ensure breathe is activated (so the API is
  listed) and that the version is embedded correctly.  [TODO: We should add a
  documentation pipeline that builds and uploads to GitHub pages.]
- Update the Spack recipe for Celeritas with the new version and sha256 value
  (either manually or using ``spack checksum``) and submit a pull request to
  the Spack project.


