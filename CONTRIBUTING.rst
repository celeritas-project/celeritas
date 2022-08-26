.. Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

Contributing to Celeritas
=========================

Thank you for your interest in the Celeritas project! Although Celeritas is
developed primarily by a group of U.S. Department of Energy subcontractors, our
goal is to increase community involvement and integration over time. We welcome
your contributions!

.. note:: This document is a work in progress.

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

All tests must pass before a PR can be merged. (Unless the failure are clearly unrelated 
to the changes and enough tests and/or configuration are passing to show the PR code
is working.  For example some of the configuration currently fail due to disk space issues.)

Since there are few enough merge requests these days, only Maintainers may
commit a merge.

Ownership and Authorship
------------------------

The person who writes a line of code is its author. However, Celeritas is a
collaborative project with collective ownership: as much as possible, there
should be a shared responsibility for the code.
