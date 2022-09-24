.. Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

Contributing to Celeritas
=========================

Thank you for your interest in the Celeritas project! Although Celeritas is
developed primarily by a group of U.S. Department of Energy subcontractors, our
goal is to increase community involvement and integration over time. We welcome
your contributions!


Copyright
---------

All submissions to the Celeritas project are automatically licensed under the
terms of [the project copyright](COPYRIGHT) as formalized by the [GitHub terms
of service](https://docs.github.com/en/github/site-policy/github-terms-of-service#6-contributions-under-repository-license).


Development prerequisites
-------------------------

To meet the :ref:`formatting` requirements described in the development guide,
make sure that `clang-format`_ is installed on your development
Run ``scripts/dev/install-commit-hooks.sh`` to to install a git post-commit hook
that will amend each commit with clang-format updates if necessary.

A mostly consistent set of decorations (separators, Doxygen comment structure,
etc.) are used throughout the code, so try to make new files look like existing
ones. Use the ``celeritas-gen.py`` script (in the ``scripts/dev`` directory) to
generate skeletons for new files, and use existing source code as a guide to
how to structure the decorations.

.. _clang-format: https://clang.llvm.org/docs/ClangFormat.html


Pull request process
--------------------

Celeritas uses the "squash and merge" process to ensure continuity of the code
history and provide easy bisecting because all commits pass all tests.
Squashing eliminates the potential of broken commits and relieves developers of
the burden of worrying about clean commit messages within a branch.

Each pull request must be reviewed by at least one person knowledgable about
the section of code being modified. Physics code should be compared against
reference papers and other codes such as Geant4. By the end of the code review,
the reviewer should understand the code well enough to maintain it (by
extension or modification) in the future. The review process must be based on
*constructive feedback* ("here's a suggestion to make this better" or "did you
consider what would happen if X?"), not *destructive feedback* ("this code is
ugly").

Reviews should not be started until the "draft" status has been removed (if it
was set to begin with). Once a pull request is under review, *do not* rebase,
squash, or otherwise alter the branch history. It causes GitHub to lose
comments and causes notifications to redirect. Additionally it will lose
information about whether the test passed and failed on previous commits.

The code guidelines (see the developer section of the user manual) must be
followed for all new code and code changes. This includes the use of the
correct formatting as well as the addition of new unit tests for new code or
bug fixes.

All tests must pass on the CI runner before a PR can be merged. (Exceptions
will be made if any failures are clearly unrelated to the changes and enough
tests and/or configuration are passing to show that the new code is working.
For example, some of the configurations currently fail due to disk space issues.)

Since there are few enough merge requests these days, only Maintainers may
commit a merge.


Ownership and Authorship
------------------------

The person who writes a line of code is its author but not its owner.
Celeritas is a collaborative project with collective ownership: as much as
possible, there should be a shared responsibility for the code.
If the code is reviewed according to the guidelines above, at least two people
should always be comfortable modifying any piece of code.
