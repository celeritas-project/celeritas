.. Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _administration:

**************
Administration
**************

This appendix includes administrative details describing the roles and
responsibilities of participants. It is meant to be a living document, more
descriptive than prescriptive. It should be a basis for future decision making,
using past decisions as a prior. Changes to this appendix can be made by pull
request with the leadership team as reviewers.


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
   The "leadership team" comprises the PIs and capability leads in the PMP, or
   their designated proxies who work directly on the Celeritas code. They are
   responsible for the long-term and big-picture project goals, and they ensure
   that project milestones are met.

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
   contribute new code, perform code reviews, publish new results using
   Celeritas, and/or participate in Celeritas stand-up meetings and hackathons.

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


.. _code_review:

Code review
===========

Reviewing incoming code maintains (and should improve) the quality of the
codebase in the interests of correctness and long-term maintainability.
If the code is reviewed according to the guidelines below, at least two people
(the author and the reviewer) will be able to modify any given piece of code,
increasing the `bus factor`_.

.. _bus factor: https://en.wikipedia.org/wiki/Bus_factor

Review process
--------------

Each pull request must be reviewed by at least one
member of the :ref:`core team <roles>` who is knowledgeable about
the section of code being modified.

The review process must be based on
*constructive feedback* ("here's a suggestion to make this better" or "did you
consider what would happen if X?"), not *destructive feedback* ("this code is
ugly"). When reviewing, you should focus almost exclusively on the new
changeset, as opposed to larger systemic issues, or smaller problems with
nearby sections of code that you happen to notice. If you do find such issues
and they can reasonably be bundled into the submitted change set, you can work
with the author to incorporate the changes, create a follow-on pull request
yourself, or open an issue for later.

Try to fully review the entirety of the changeset on your first review, *but*
if major changes are needed it's a good idea to pause, submit your comments so
far, and work with the PR author to address the major issues before continuing
to the next review iteration.

The correctness of the new code must be ensured by comparing against
references, verifying the new code is sufficiently tested, and/or running
regression problems on the incoming branch.
Physics code should be compared against reference papers and other codes such
as Geant4.

Ensure readability and maintainability by checking that the :ref:`code
<code_guidelines>` and :ref:`style <style_guidelines>` guidelines have been
followed in the new code. Balance the desire for readability with the need to
avoid bikeshedding_ by asking yourself whether your requests are
substantive enough to merit a new pull request. Perfect is the enemy of good.

By the time you've finished the code review, you should understand the code
well enough to maintain it (by extension or modification) in the future.

.. _bikeshedding: https://thedecisionlab.com/biases/bikeshedding


Merge process
-------------

Celeritas uses the "squash and merge" process to ensure continuity of the code
history and provide easy bisecting because all commits pass all tests.
Squashing eliminates the potential of broken commits and relieves developers of
the burden of worrying about clean commit messages within a branch.

Since there are few enough merge requests these days, only :ref:`maintainers
<roles>` may commit a merge.


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


Releases can be created from the primary "develop" branch (major, minor, patch)
or a "backport" branch (minor, patch).
The following process must be followed (and may need iteration to converge) for
each release.

- Create a ``release-vX.Y.Z`` branch.
- Ensure all CI jobs passed for the release in question. This is automatic for
  releases from the ``develop`` branch (since every pull request must pass) but
  should
  be checked manually for backports.
- Run regression tests on Summit (for performance testing), Crusher (for HIP
  testing), and an additional machine with debug assertions enabled (e.g.
  Wildstyle). Postpone the release
  temporarily if major new bugs or performance regressions are detected. If
  minor updates are needed to fix the build or tests on a particular machine,
  include those as part of the "pre-release" pull request that includes new
  documentation.
- [TODO: define high-level validation tests like `geant-val`_ and a test matrix
  correlating capability areas (code files/directories changed) to test names.]
  Rerun and perform a cursory check on all validation tests that might be
  affected by changes since the previous release. More complete validation
  (since a change in results might not be an error) can be done separately.
- Update documentation with release notes from all pull requests newly included
  in the release. Follow the format for previous releases: add a summary of
  highlights, and enumerate the pull requests (with PR numbers and
  authorship attribution) separated by features and bug requests. Use the
  `helper notebook`_ in the Celeritas documents repository to automate this.
- Ensure the code documentation builds, preferably without warnings, on a
  configuration that has Sphinx, Doxygen, and Breathe active. [TODO: automate
  this with CI for doc publishing]
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

The first commit that deviates from the most recent major or minor branch
should be tagged (but not released!) with the next version number with a
``-dev`` suffix. For example, after releasing version 1.0.0, the next
commit on the ``develop`` branch that is *not* intended for version 1.0.1
(i.e., the
first new feature) should be tagged with ``v1.1.0-dev``, so that
``git describe --tags --match 'v*'`` shows the new features as being part of the
``v1.1.0`` series.

.. _helper notebook: https://github.com/celeritas-project/celeritas-docs/blob/master/nb/admin/github-stats.ipynb
.. _geant-val: https://geant-val.cern.ch
