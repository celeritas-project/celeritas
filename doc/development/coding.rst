.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. **NOTE**: this file is referenced by README.md:
.. if changing the former, update the latter!!

.. _code_guidelines:

Code development guidelines
===========================

Every new piece of code is a commitment for you and other developers to
maintain it in the future (or delete it if obsolete). There are numerous
considerations to making code easier to update or understand, including testing
and documentation.


Document implicitly and explicitly
----------------------------------

Code should be self-documenting as far as possible (see details below for
naming conventions). This means that variable names, function names, and
function arguments should be as "obvious" as possible. Take particular care
with constants that appear in physics implementations. They should
be multiplied by units in the native Celeritas unit system if applicable, or
defined as ``Quantity`` instances. The numerical value of the constant must
also be documented with a paper citation or other comment.

Class documentation through Doxygen (see :ref:`formatting`) can be injected
semi-automatically into this user manual via the Breathe tool integrated
into the Celeritas build system (see :ref:`dependencies`). High-level classes
should describe the functionality of the class in a way understandable to both
power users and developers, and such classes should be included in the
:ref:`api` section. Inline ReStructured Text can be added to the user manual
using the custom aliases ``\rst`` and ``\endrst``.

Citations to the Celeritas Zotero database can be referenced inline code using
the special ``\citep{whatever-2025, https://doi.org/.../}`` Doxygen
alias set up inside the Celeritas doc :file:`CMakeLists.txt`. (Another form,
``\citet``, will insert the reference "textually" as opposed to
"parenthetically"; and use the single-argument ``\cite`` if you do not have a
DOI.) The procedure to set up a citation correctly is:

#.  First, set up `Better BibTeX for Zotero`_, and in the Zenodo settings under
    "Better BibTeX," import the preferences at
    :file:`scripts/dev/better-bibtex-prefs.json`.
#.  Ensure you have access to the `Celeritas group`_ library; contact
    @tmdelellis if you are not a member of the group.
#.  Add the reference to the appropriate subdirectory of the Zotero Celeritas
    References subgroup.  If you have a DOI, use the handy magic wand icon to
    "Add items by identifier".
#.  Optionally, in the "Extra" field of the resulting item, add a special line:
    ``Citation key: author-keyword-year`` to set up a stable BibTeX citation
    key. The default citation key is ``author-shorttitle-year``.
#.  Export the "Celeritas" group using the "Better BibTeX" format, without
    notes, files, or journal abbreviations. Save the file to
    :file:`doc/_static/zotero.bib`.

.. _Better BibTeX for Zotero: https://github.com/retorquere/zotero-better-bibtex
.. _Celeritas group: https://www.zotero.org/groups/2380941/celeritas/library

.. _coding_testing:

Test thoroughly
---------------

Functions should use programmatic assertions whenever assumptions are made.
Celeritas provides three assertions

- Use the :c:macro:`CELER_EXPECT` assertion macro to test preconditions about
  incoming data or initial internal states.
- Use :c:macro:`CELER_ASSERT` to express an assumption internal to a function (e.g.,
  "this index is not out of range of the array").
- Use :c:macro:`CELER_ENSURE` to mark expectations about data being returned from a
  function and side effects resulting from the function.

These assertions are enabled only when the ``CELERITAS_DEBUG`` CMake option is
set.
Additionally, user-provided data and potentially volatile runtime conditions
(such as the presence of an environment variable) should be checked with
the always-on assertion :c:macro`CELER_VALIDATE` macro.

Each class must be thoroughly tested with an independent unit test in the
`test` directory.  For complete coverage, each function of the class must have
at least as many tests as the number of possible code flow paths (cyclomatic
complexity).

Implementation detail classes (in the ``celeritas::detail`` namespace, in
``detail/`` subdirectories) are exempt from the testing requirement, but
testing the detail classes is a good way to simplify edge case testing compared
to testing the higher-level code.


Maximize encapsulation
----------------------

Encapsulation is about making a piece of code into a black box. The fewer lines
connecting these black boxes, the more maintainable the code. Black boxes can
often be improved internally by making tiny black boxes inside the larger black
box.

Motivation:

- Developers don't have to understand implementation details when looking at a
  class interface.
- Compilers can optimize better when dealing with more localized components.
- Good encapsulation allows components to be interchanged easily because they
  have well-defined interfaces.
- Pausing to think about how to minimize input and output from an algorithm can
  improve it *and* make it easier to write.

Applications:

- Refactor large functions (> 50-ish statements?) into small functors that take
  "invariant" values (the larger context) for constructors and use
  ``operator()`` to transform some input into the desired output
- Use only ``const`` data when sharing. Non-const shared data is almost like
  using global variables.
- Use ``OpaqueId`` instead of integers and magic sentinel values for
  integer identifiers that aren't supposed to be arithmetical.

Examples:

- Random number sampling: write a unit sphere sampling functor instead of
  replicating a polar-to-Cartesian transform in a thousand places.
- Volume IDs: Opaque IDs add type safety so that you can't accidentally convert
  a volume identifier into a double or switch a volume and material ID. It also
  makes code more readable of course.

Encapsulation is also useful for code reuse. Always avoid copy-pasting code, as
it means potentially duplicating bugs, duplicating the amount of work needed
when refactoring, and missing optimizations.


Minimize compile time
---------------------

Code performance is important but so is developer time. When possible,
minimize the amount of code touched by NVCC. (NVCC's error output is also
rudimentary compared to modern clang/GCC, so that's another reason to prefer
them compiling your code.)


Prefer single-state classes
---------------------------

As much as possible, make classes "complete" and valid after calling the
constructor. Try to avoid "finalize" functions that have to be called in a
specific order to put the class in a workable state. If a finalize function
*is* used, implement assertions to detect and warn the developer if the
required order is not respected.

When a class has a single function (especially if you name that function
``operator()``), its usage is obvious. The reader also doesn't have to know
whether a class uses ``doIt`` or ``do_it`` or ``build``.

When you have a class that needs a lot of data to start in a valid state, use a
``struct`` of intuitive objects to pass the data to the class's constructor.
The constructor can do any necessary validation on the input data.


Learn from the pros
-------------------

Other entities devoted to sustainable programming have their own guidelines.
The `ISO C++ guidelines`_ are very long but offer a number of insightful
suggestions about C++ programming. The `Google style guide`_ is a little more
targeted toward legacy code and large production environments, but it still
offers good suggestions. For software engineering best practices, the
book `Software Engineering at Google`_ is an excellent reference. The `LLVM
coding standards`_ also have good guidelines for developing maintainable C++
in the context of a large project.

.. _ISO C++ guidelines: http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
.. _Google style guide: https://google.github.io/styleguide/cppguide.html
.. _Software Engineering at Google: https://abseil.io/resources/swe-book
.. _LLVM coding standards: https://llvm.org/docs/CodingStandards.html

