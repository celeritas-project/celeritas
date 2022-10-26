.. Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _contributing:

Contributing to Celeritas
=========================

Thank you for your interest in the Celeritas project! Although Celeritas is
developed primarily by a group of U.S. Department of Energy subcontractors, our
goal is to increase community involvement and integration over time. We welcome
your contributions!


Copyright
---------

All submissions to the Celeritas project are automatically licensed under the
terms of :ref:`the project copyright <code_copyright>` as formalized by the
`GitHub terms of service`_.
The person who writes a line of code is its author and copyright holder, but
the new code should be the shared responsibility of the project rather than the
exclusive property of a single contributor.
Celeritas is a collaborative project with *collective* ownership.


.. _GitHub terms of service: https://docs.github.com/en/github/site-policy/github-terms-of-service#6-contributions-under-repository-license


Collaborating
-------------

Working with other Celeritas team members is a critical part of the development
process. Please join the ``#code`` channel on the `Celeritas Slack workspace`_,
open an issue with a bug report or feature request, or start a discussion on
GitHub.

.. _Celeritas Slack workspace: https://celeritasproject.slack.com/


Development prerequisites
-------------------------

Create a fork_ of the Celeritas repository. You should clone this fork locally
to your development machine as the "origin", and it's a good idea to add the
main ``celeritas-project`` repository as an "upstream" so that you can apply
changes from the main codebase as you work.

To meet the :ref:`formatting` requirements described in the development guide,
make sure that `clang-format`_ is installed on your development machine.
Run ``scripts/dev/install-commit-hooks.sh`` to install a git post-commit hook
that will amend each commit with clang-format updates if necessary.

A mostly consistent set of decorations (separators, Doxygen comment structure,
etc.) are used throughout the code, so try to make new files look like existing
ones. Use the ``celeritas-gen.py`` script (in the ``scripts/dev`` directory) to
generate skeletons for new files, and use existing source code as a guide for
how to structure the decorations.

.. _fork: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks
.. _clang-format: https://clang.llvm.org/docs/ClangFormat.html


Submitting code changes
-----------------------

When you believe that you've made a substantive [#]_ and self-contained
improvement to the code, it's time to create a `pull request`_ (PR) to get
feedback on your changes before they're merged into the code base.

Before opening the pull request, check that the :ref:`code <code_guidelines>`
and :ref:`style <style_guidelines>` guidelines have been followed for all new
code and code changes.  Ensure the use of the correct formatting as well as the
addition of documentation and unit tests for new code and bug fixes.

All tests must pass on the CI runner before a PR can be merged. (Exceptions
will be made if any failures are clearly unrelated to the changes and enough
tests and/or configuration are passing to show that the new code is working.
For example, some of the configurations have a tendency to fail due to disk
space issues.) It's best to test locally first before submitting your pull
request, and keep in mind that the multiple configurations on the CI (different
dependency versions, different features) may reveal failures that your local
testing might have missed.

Each pull request should be assigned one or two reviewers who will provide
constructive feedback on your changes. Their responsibilities are outlined in
:ref:`code_review`.
Reviews should not be started until the "draft" status has been removed (if you
marked it as a draft initially). Once a pull request is under review, *do not*
rebase, squash, or otherwise alter the branch history. Such changes can
drastically increase the difficulty of reviewing, because it may blend in a
single commit both changes in response to a review *and* changes from upstream
code. (Furthermore, it breaks GitHub notifications and makes it more difficult
to find older comments.)  You *can* merge the main upstream branch if
your changes may interact with the upstream changes, and you *must* merge if
they conflict.

After the CI tests pass and the reviewers have approved your changes,
your branch will be squashed and merged, and you will officially be a Celeritas
:ref:`contributor <roles>`! Congratulations!

.. [#] All changes to the codebase must go through the pull request, but due to
   the overhead of reviewing, testing, merging, and documenting a PR, we'd like
   to avoid small changes that have almost no effect in terms of operation or
   readability. For example, if you find a typo in the documentation, check the
   rest of the docs for any other typos or improvements you'd like to make, and
   submit a single PR with those changes.

.. _pull request: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests
