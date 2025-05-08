.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _style_guidelines:

Style guidelines
================

Having a consistent code style makes it more readable and maintainable. (For
example, you don't have to guess whether a symbol is a function or class.)

As a historical note, many of the style conventions in Celeritas derive from
the `Draco project style`_ of which `Tom Evans`_ was primary author and which
became the style standard for the GPU-enabled Monte Carlo code `Shift`_.

.. _Draco project style: https://github.com/lanl/Draco/wiki/Style-Guide
.. _Tom Evans: https://github.com/tmdelellis
.. _Shift: https://doi.org/10.1016/j.anucene.2019.01.012


.. _formatting:

Formatting
----------

Formatting is determined by the clang-format file inside the top-level
directory. One key restriction is the 80-column limit, which enables multiple
code windows to be open side-by-side. Generally, statements longer than 80
columns should be broken into sub-expressions for improved readability anyway
-- the ``auto`` keyword can help a lot with this. The `pre-commit`_ utility
(execute ``pre-commit install --install-hooks`` or run
:file:`scripts/dev/install-commit-hooks.sh`) will take
care of formatting automatically. Clang-format will also
enforce the use of "`East const`_", where the ``const`` keyword is always to
the right of the type that it modifies.

Certain decorations (separators, Doxygen comment structure,
etc.) are standard throughout the code. Use the :file:`celeritas-gen.py` script
(in the :file:`scripts/dev` directory) to generate skeletons for new files, and
use existing source code as a guide to how to structure the decorations.

.. _pre-commit: https://pre-commit.com
.. _East const: https://hackingcpp.com/cpp/design/east_vs_west_const.html


Documentation
-------------

Doxygen comments should be provided next to the *definition* of functions (both
member and free) and classes. This means adding a one-line Doxygen comment for
member functions defined inside the class's definition or multi-line Doxygen
comments if a function is defined externally.
Document the effect of a function-like class's "call" operator``()`` in the class's main definition rather than the operator
itself, since this makes it easier and cleaner to document the class's behavior
in the :ref:`api` documentation. Do the same for physics classes.


Symbol names
------------

Functions should be verbs; classes should be names. As in standard Python
(PEP-8-compliant) code, classes should use ``CapWordsStyle`` and variables use
``snake_case_style``. A symbol should have a trailing underscore *always and
only* if it is private member data: neither public member data nor private
member functions should have them.

Functors (classes whose instances act like a function) should be an *agent
noun*: the noun form of an action verb. Instances of a functor should be a
verb. For example::

   ModelEvaluator evaluate_something(parameters...);
   auto result = evaluate_something(arguments...);

There are many opportunities to use ``OpaqueId`` in GPU code to indicate
indexing into particular vectors. To maintain consistency, we let an
index into a vector of ``Foo`` objects have a corresponding ``OpaqueId``
type::

    using FooId = OpaqueId<Foo>;

and ideally be defined either immediately after ``Foo`` or in a
:file:`Types.hh` file.  Some ``OpaqueId`` types may have only a "symbolic"
corresponding type, in which case a tag struct can be be defined inline, using
an underscore suffix as a convention indicating the type does not correspond to
an actual class::

   using BarId = OpaqueId<struct Bar_>;

.. note:: Public functions in user-facing Geant4 classes (those in ``accel``)
   should try to conform to Geant4-style naming conventions, especially because
   many will derive from Geant4 class interfaces.


File names
----------

We choose the convention of ``.cc`` for C++ translation units and
corresponding ``.hh`` files for C++ headers.

Thus we have the following rules:

- ``.hh`` is for C++ header code compatible with host compilers. The code in
  this file can be compatible with device code if it uses the
  ``CELER_FUNCTION`` family of macros defined in ``corecel/Macros.hh``.
- ``.cc`` is for C++ code that will invariably be compiled by the host
  compiler.
- ``.cu`` is for ``__global__`` kernels and functions that launch them,
  including code that initializes device memory. Despite the filename, these
  files should generally also be HIP-compatible using Celeritas abstraction
  macros.
- ``.device.hh`` and ``.device.cc`` require CUDA/HIP to be enabled and use the
  library's runtime libraries and headers, but they do not execute any device
  code and thus can be built by a host compiler. If the CUDA-related code is
  guarded by ``#if CELER_USE_DEVICE`` macros then the special extension is not
  necessary.

Some files have special extensions:

- ``.t.hh`` is for non-inlined ``template`` implementations intended to be
  included and explicitly instantiated in a downstream C++ or CUDA compilation
  unit.  Note that if the function in the ``.hh`` file is declared ``inline``,
  the definition must be provided in the header as well.
- ``.test.cc`` are unit test executables corresponding to the main ``.cc``
  file. These should only be in the main ``/test`` directory.


.. _device_compilation:

Device compilation
------------------

All ``__device__`` and ``__global__`` code must be compiled with NVCC or
HIPCC to generate device objects. However, code that merely uses CUDA API calls
such as
``cudaMalloc`` does *not* have to be compiled with NVCC. Instead, it only has to
be linked against the CUDA runtime library and include ``cuda_runtime_api.h``.
The platform-agnostic Celeritas include file to use is
``corecel/DeviceRuntimeApi.hh``.
Note that VecGeom compiles differently when run
through NVCC (macro magic puts much of the code in a different namespace) so
its inclusion must be very carefully managed.

Since NVCC is slower and other compilers' warning/error output is more
readable, it's preferable to use NVCC for as little compilation as possible.
Furthermore, not requiring NVCC lets us play nicer with downstream libraries
and front-end apps. Host code will not be restricted to the maximum C++ standard version
supported by NVCC.

Of course, the standard compilers cannot include any CUDA code containing
kernel launches, since those require special parsing by the compiler. So kernel
launches and ``__global__`` code must be in a ``.cu`` file. However, the
CUDA runtime does define the special ``__host__`` and ``__device__`` macros (among
others). Therefore it is OK for a CUDA file to be included by host code as long
as it ``#include`` s the CUDA API. (Note that if such a file is to be included by
downstream code, it will also have to propagate the CUDA include directories.)

Choosing to compile code with the host compiler rather than NVCC also provides
a check against surprise kernel launches. For example, the declaration::

   thrust::device_vector<double> dv(10);

actually launches a kernel to fill the vector's initial state. The code will
not compile in a ``.cc`` file run through the host compiler, but it will
automatically (and silently) generate kernel code when run through NVCC.


Variable names
--------------

Generally speaking, variables should have short lifetimes and should be
self-documenting. Avoid shorthand and "transliterated" mathematical
expressions: prefer ``constants::na_avogadro`` to ``N_A`` (or express the
constant functionally with ``atoms_per_mole``) and use ``atomic_number``
instead of ``Z``. Physical constants should try to have the symbol concatenated
to the context or meaning (e.g. `c_light` or `h_planck`).

Use scoped enumerations (``enum class``) where possible (named like classes) so
their values can safely be named like member variables (lowercase with
underscores). Prefer enumerations to boolean values in function interfaces
(since ``do_something(true)`` requires looking up the function interface
definition to understand).


Function inputs and outputs
---------------------------

The following rules are mostly derived from the `Google style guide`_, so refer
to that reference if not specified here.

- Always pass value types for arguments when the data is small (``sizeof(arg)
  <= sizeof(void*)``). Using values instead of pointers/references allows the
  compiler to optimize better. If the argument is nontrivial but you need to
  make a local copy anyway, it's OK to make the function argument a value (and
  use ``std::move`` internally as needed, but this is a more complicated
  topic).
- Use const *references* for types that are nontrivial and that you only need
  to access or pass to other const-reference functions.
- Prefer return values or structs rather return-by-reference. This
  makes it clear that there are no preconditions on the input value's state.
- In Celeritas we *formerly* used the google style of passing mutable pointers
  instead of mutable references, so that it's more obvious to the calling code
  that a value is going to be modified. The Google style changed and this has
  fallen out of favor; **USE REFERENCES** except for the very rare case of
  *optional* return values.
- Host-only (e.g., runtime setup) code should almost never return raw pointers;
  use shared pointers instead to make the ownership semantics clear. When
  interfacing with older libraries such as Geant4, try to use ``unique_ptr``
  and its ``release``/``get`` semantics to indicate the transfer of pointer
  ownership.
- Avoid ``const`` *values* (e.g. ``const int``), because the decision
  to modify a local variable or not is an implementation detail of the
  function, not part of its interface. Clang-tidy will warn about this.

.. _Google style guide: https://google.github.io/styleguide/cppguide.html#Inputs_and_Outputs

Memory is always managed from host code, since on-device data management can be
tricky, proprietary, and inefficient. There are no shared or unique pointers,
but there is less of a need because memory management semantics are clearer.
Device code has exceptions from the rules above:

- In low-level device-compatible code (such as a ``TrackView``), it is OK to
  return a pointer from a function to indicate that the result may be undefined
  (i.e., the pointer is null) or a non-owning **reference** to valid memory.
  This is used in the ``StackAllocator`` to indicate a failure to allocate new
  memory, and in some accessors where the result is optional.
- The rule of passing references to complex data does not apply to CUDA
  ``__global__`` kernels, because device code cannot accept references to host
  memory. Instead, kernel parameters should copy by value or provide raw
  pointers to device memory. Indicate that the argument should not be used
  inside the kernel can prefix it with ``const``, so the CUDA compiler places
  the argument in ``__constant__`` memory rather than taking up register space.


Polymorphism and virtual functions
----------------------------------

Since polymorphism on GPUs incurs severe performance and infrastructure
penalties, virtual functions *must* be limited to host-only setup and runtime
functions. If at all possible, follow these guidelines:

- Use only pure abstract virtual classes if possible (no methods should be
  defined; all methods should be ``virtual ... = 0;``). Instead of adding helper
  functions or protected data, use *composition* to define such things in a
  separate class.
- If the abstract class is to be used in downstream code, `define an
  out-of-line function to reduce potential code bloat
  <https://stackoverflow.com/questions/12024642/placing-of-external-virtual-tables/12025816#12025816>`.
- Use public virtual destructors to allow base-class deletion (e.g., a
  ``unique_ptr`` to the base class) *or* use a protected nonvirtual destructor
  if the classes are not meant to be stored by the user.
- Define protected ``CELER_DEFAULT_COPY_MOVE`` constructors to prohibit
  accidental operations between base classes.

In daughter classes:

- Prefer daughter classes to implement all of the functionality of the base
  classes; this makes it easier to reason about the code because all the
  operations are local to that implementation.
- Use the ``final`` keyword on classes *except* in the rare case that this
  class is providing new extensible interfaces.
- Use exactly one of the ``final`` or ``override`` keywords for inherited
  virtual functions. Most classes should only have "final" methods.


Odds and ends
-------------

Although ``struct`` and ``class`` are interchangeable for class definitions
(modifying only the default visibility as public or private), use ``struct``
for data-oriented classes that don't declare constructors and have only
public data; and use ``class`` for classes designed to encapsulate
functionality and/or data.

With template parameters, ``typename T`` and ``class T`` are also
interchangeable, but use ``template <class T>`` to be consistent internally and
with the standard library. (It's also possible to have ``template <typename``
where ``typename`` *doesn't* mean a class: namely,
``template <typename U::value_type Value>``.)

Use ``this->`` when calling member functions inside a class to convey that the
``this`` pointer is implicitly being passed to the function and to make it
easier to differentiate from a free function in the current scope.
