.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. **NOTE**: this file is referenced by README.md:
.. if changing the former, update the latter!!

.. _administration:

**************
Administration
**************

This appendix includes administrative policies and details describing the roles
and responsibilities of participants. It is meant to be a living document, more
descriptive than prescriptive. It should be a basis for future decision making,
using past decisions as a prior. Changes to this appendix can be made by pull
request with the leadership team as reviewers.


Community standards
===================

Be respectful. Communicate openly. Resolve disagreements peacefully.


Resolution process
------------------

If a disagreement about Celeritas can't be immediately resolved, it's OK to
postpone resolution (maybe you need more data or more time to think, perhaps to
decide it's not important enough at the end of the day). In the case that a
resolution is
needed quickly (e.g., if it's a blocking issue), it should be discussed among
the :ref:`core team <roles>` until an agreement is reached. If the issue is
still contentious, the leadership team should make an executive decision,
giving weight to the appropriate capability area lead if possible.


.. _roles:

Roles
=====

The roles of the Celeritas code base are related to the roles
set out in the SciDAC project management plan. These roles should be
reflected in the different teams and/or access restrictions in GitHub.

Code lead
   The code lead is responsible for steering the technical implementation of
   Celeritas to meet the long-term project goals. One key responsibility of
   the code lead is to set up milestones for releases and execute the release
   process described below.

Maintainer
   Maintainers should be familiar with most or all parts of the codebase and
   are responsible for merging pull requests and closing issues.

Core team member
   The "core team" are those currently funded to work on Celeritas. Core team
   members are responsible for reviewing pull requests in accordance with the
   :ref:`contributing guidelines <contributing>`. They should regularly
   contribute new code, perform code reviews, publish new results using
   Celeritas, and/or participate in Celeritas stand-up meetings and hackathons.

Core advisor
   Advisors maintain close ties to Celeritas but are not consistently
   developing or validating it. They should be officially part of a Celeritas
   proposal or funded work even though they may charge only a small fraction of
   their time. Core advisors are encouraged to perform code reviews and attend
   meetings, and they are expected to have a leadership role in long-term
   project planning.

Contributor
   Anyone can submit a pull request that conforms to the contribution
   guidelines! Contributors are recognized by including their handle and pull
   requests in release notes but do not otherwise have a formal role.


Role change process
-------------------

Roles should be periodically reevaluated to reflect current participation in
the project. It is recommended to re-evaluate core roles at the start of each
calendar year based on the last year's participation and upcoming work plans.

Adding or removing a member of the "core team" can be done at any time by
consensus of the
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

Reviewing pull requests
-----------------------

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

Check that the title meets the requirements below, that the description is
adequate, and that the appropriate labels are set.

Mark your comments as resolved only if the other person was the last to
comment. Other reviewers can reopen the conversation to comment on it. All
conversations must be resolved for the merge to be approved.

By the time you've finished the code review, you should understand the code
well enough to maintain it (by extension or modification) in the future.

.. _bikeshedding: https://thedecisionlab.com/biases/bikeshedding

PR titles
^^^^^^^^^

The title should be an imperative statement (title-cased first word,
no trailing punctuation) summarizing its effect on the user.  For example:

 - Implement the FooBar model *[enhancement, physics]*
 - Handle errors in track initialization *[enhancement, orange]*
 - Fix sampling of low-energy Celeritons *[bug, physics]*
 - Refactor code in preparation for new tracker type *[minor, orange]*
 - Add CI support for multiple Geant4 versions *[enhancement, documentation]*

Avoid tags in the title.

PR description
^^^^^^^^^^^^^^

The description should summarize or enumerate the main changes in the pull
request. Illustrative images are recommended if possible!

Label descriptions
^^^^^^^^^^^^^^^^^^

The labels_ used in Celeritas GitHub pull requests and issues have specific
meanings. There should be one "change type" and one "category" selected. A few
special labels ("backport", "performance") can also be added when appropriate.

.. _labels: https://github.com/celeritas-project/celeritas/labels

Change type
"""""""""""

The change types are used to categorize changes in the release notes. If a pull
request seems like it should have multiple change types, it should probably be
broken up into multiple pull requests.

Bug
   Report or fix an error that a user could encounter.

Documentation
   Add new tests, documentation, or personal user presets *without* any changes
   in the library or applications.

Enhancement
   Request or add something new to the code.

Minor
   Refactor code or make small changes that should be effectively invisible to
   users and probably not reported.

Category
""""""""

These should be fixes, changes, or enhancements to support...

App
   front end applications (``celer-g4`` and ``celer-sim``).

Core
   core infrastructure such as platform portability.

External
   integration with external libraries such as Geant4, VecGeom, and ROOT; or
   requirements from experiment frameworks.

Field
   magnetic field propagation, linear propagation, and safety distance
   calculation.

ORANGE
   the Celeritas geometry library for GPU tracking.

Performance
   performance optimization on GPU and CPU.

Physics
   particles, processes, and stepping algorithms.

User
   hits and track diagnostics (i.e., extracting data from tracks running on the
   GPU).


Approving
---------

Reviews on GitHub can be submitted with three different outcomes:

Request changes
  This is like "reject unless revised" for a manuscript: there are significant
  concerns that need to be fixed/addressed before it can be merged, and this
  status *must* be cleared before merging.

Comment
  Comments should be added if there are minor issues, questions, or suggestions
  that may be significant enough to need addressing before merge.

Approve
  Once the review process above has been completed successfully, the merge
  request should be approved. As a courtesy, if there are only mild questions
  or comments that the reviewer believes to need no further discussion, the PR
  can be marked as "approved" while the submitter addresses those questions.

Merging
-------

The GitHub settings for Celeritas currently require that before merging:

1.  The branch must be up-to-date with upstream develop. This ensures that
    there are no failures from "implicit" conflicts (where no code actually
    generates a git conflict, but some new requirement or change upstream
    has not been implemented in the new branch). An "Update branch" button is
    available on the pull request page to ensure this requirement is met.
2.  The GitHub Action CI checks must pass. There are a small matrix of core
    combinations that cover most potential build issues. These do not *execute*
    GPU code but they will build it and ensure that the tests pass when
    the Celeritas device capability is disabled.
3.  All conversations must be resolved (see the "reviewing" section above and
    the contributing guidelines).

Celeritas uses the "squash and merge" process to ensure continuity of the code
history and provide easy bisecting because all commits pass all tests.
Squashing eliminates the potential of broken commits and relieves developers of
the burden of worrying about clean commit messages within a branch.

Since there are few enough merge requests these days, only :ref:`maintainers
<roles>` may commit a merge. When merging, check that the commit title matches
the issue title (it may be inconsistent if the branch has only a single
commit), and that the "co-author" tags at the bottom of the commit message
accurately reflect contributions (co-authorship may be erroneously attributed
for a simple merging of the main branch into the development branch).


.. _documentation:

Documentation
=============

The documentation for Celeritas is maintained alongside the main source
repository in the :file:`doc` subdirectory. It should be kept up-to-date as
part of each pull request: new or changed physics must be reflected as part of
that PR, for example. The primary place for documentation should be the user
docs, and the most important information should be in the main ``README.md`` in
the repository.

A Github workflow updates the
official `github pages`_ site on every push to ``develop`` by generating the
developer and user documentation with Doxygen and Sphinx. The "static" part of
the web site is modifiable by using the ``doc/gh-pages-base`` branch. This
branch contains an index file that points to the documentation; it also
includes auto-generated publications and references and the script to generate
them.
Additionally there are two special repositories in the ``celeritas-project``
organization that are used for documentation and should also just be redirects.

In summary, there are several possible sources of documentation, most of which
should be simple redirects to reduce maintenance:

- ``https://github.com/celeritas-project`` : a top-level organization README
  generated by a special `.github <https://github.com/celeritas-project/.github>`_ repository
- ``https://github.com/celeritas-project/celeritas`` : a code README generated
  by the ``README.md`` in the main Celeritas repo
- ``https://celeritas-project.github.io``: a special organization-level
  `github.io <https://github.com/celeritas-project/celeritas-project.github.io>`_
  repository that should just redirect to the useful documentation
- ``https://celeritas-project.github.io/celeritas``: generated from the
  ``doc/gh-pages-base`` branch and the Celeritas Zotero list
- ``https://celeritas-project.github.io/celeritas/dev``: developer docs
  generated from Doxygen via the ``develop`` branch (see the
  ``doc/CMakeLists.txt`` file)
- ``https://celeritas-project.github.io/celeritas/user``: user docs generated
  from Sphinx via the ``develop`` branch (see ``doc/index.rst``)

.. _github pages: https://celeritas-project.github.io/celeritas/


Releases
========

Celeritas uses `Semantic Versioning`_ to enumerate releases. During its initial
development phase, ``0.x.0`` is a major release and ``0.x.z`` is a patch
release. When Celeritas is declared stable, ``x.0.0`` is a major release,
``x.y.0`` is a minor release, and ``x.y.z`` is a patch release.

Major and minor releases (including 0.x.0 development releases) must have a
milestone in the git issue tracker with a list of issues that can be assigned.
Only major releases can remove features and change
:ref:`public-facing APIs <api>`. Minor releases can
:ref:`deprecate features <deprecations>`. Both major and minor releases should
include notable improvements to the code.

Patch releases can be created at any time but should typically include at least
one critical bug fix or several substantial fixes. Patch releases should focus
almost exclusively on fixes and should generally not include new features or
other major code changes.

.. _Semantic Versioning: https://semver.org

Release process
---------------

.. ***NOTE*** when changing this section, make sure
   ``.github/pull_request_template.release`` is also updated.

Releases can be created from the primary "develop" branch (major, minor, patch)
or a "backport" branch (minor, patch).
The following process must be followed (and may need iteration to converge) for
each release.

#.  Ensure all CI jobs pass for the target branch to be released (develop or
    backports/vX.Y). This is automatic
    for releases from the ``develop`` branch, since every pull request must
    pass, but should be checked manually for backports.
#.  Create a ``release-vX.Y.Z`` branch from *upstream/develop*.
#.  If creating a new release from develop, tag the target branch with
    ``vX.Y.Z-rc.N`` where N starts with 1, and
    increment for every time you return to this step due to new pull requests.
    The tag can be pushed to your fork, or to the main repository if it should
    be shared with other team members.
#.  Run performance regression tests on Perlmutter (for performance
    testing), Frontier (for HIP testing), and an additional machine
    with debug assertions enabled (e.g., Wildstyle).
#.  [TODO: define high-level validation tests like `geant-val`_ and a test
    matrix correlating capability areas (code files/directories changed) to
    test names.] Rerun and perform a cursory check on all validation tests that
    might be affected by changes since the previous release. More complete
    validation (since a change in results might not be an error) can be done
    separately.
#.  Postpone the release temporarily if major new bugs or performance
    regressions are detected. Create new pull requests for the serious errors
    using the standard :ref:`contributing <contributing>` process, and once the
    fixes are merged into develop, merge develop into the release branch.
    Return to step 2.
#.  If only minor updates are needed to fix the build or tests on a particular
    machine, include those as part of the release branch.
#.  If this is a "major" release (see :ref:`deprecations`),
    check for and remove code marked as ``DEPRECATED: to be removed
    in vX.Y``.
#.  Update documentation with release notes from all pull requests newly
    included in the release. *Make sure this happens after all pull requests
    targeted for this milestone have been merged*.
    Follow the format for previous releases: add a
    summary of highlights, and enumerate the pull requests (with PR numbers and
    authorship attribution) separated by features and bug requests. Use the
    `helper notebook`_ in the Celeritas documents repository to automate this.
    Ensure that both the HTML and PDF versions of the documentation build
    without errors.
#.  Submit a pull request with the newly added documentation and any
    release-related tweaks using the
    :file:`.github/pull_request_template.release.md` template, and wait until
    it's reviewed and merged. The unit tests and documentation should all build
    and pass the CI.
#.  If releasing a backported version branch, cherry-pick this documentation
    commit into the backport branch.
#.  Use the GitHub interface to create a new release with the documentation
    update that was just added.

After committing the release tag:

#. Save the ``tar.gz`` and attach to the release, because the hash changes if
   the git "describe" function returns a different result for the release tag's
   hash (e.g., if a collaborative branch on the main repository points to that
   commit).
#. Pull locally (make sure to use the ``--tags`` option) and build PDF user
   documentation for the release. Ensure breathe is activated (so the API is
   listed) and that the version is embedded correctly.  [TODO: We should add a
   documentation pipeline that builds and uploads to GitHub pages.]
#. Update the Spack recipe for Celeritas with the new version and sha256 value
   (either manually or using ``spack checksum``) and submit a pull request to
   the Spack project.
#. Mark the GitHub release milestone as completed.

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

.. _deprecations:

Deprecations
------------

Deprecating obsolete code is vital to the long-term maintainability of an
open-source project. As new capabilities and better interfaces replace old
ones, removing the old ones is the only way to pay off technical debt. A
careful deprecation process is necessary to provide users a way to transition
to the newer capabilities: there must be separate releases marking code as
deprecated and removing it, and removal is only allowed in major version
changes.

Deprecated public APIs (functions, classes, identifiers, ...) should be marked
in the code with the ``[[deprecated]]`` C++ attribute and an adjacent comment
"remove in vX.0". Here, X is the next major release after the deprecation is
released [#]_. For example, if a function is deprecated after version 1.2 is
released but a 1.3 release is planned, the comment should specify ``remove in
v2.0``. However, if the deprecation is made after the final minor version is
released (i.e., on or after the ``v2.0-dev`` tag) the deprecation should be
marked for ``v3.0``.

Private APIs (those not documented in the user API documentation, *not* limited
to classes in the ``detail`` namespace) are not subject to the deprecation
policy and can be changed at will. As the Celeritas code and its use cases
mature, some functionality will become public and others will become "private."
Making a public API private should be treated as a deprecation.

.. [#] During initial development, deprecations will target ``v0.Y``.
