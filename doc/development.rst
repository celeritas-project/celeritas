=====================
Celeritas development
=====================


Core guidelines
===============

In which the author writes his first manifesto since probably high school.

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
  improve make it easier to write.

Applications:

- Refactor large functions (> 50 statements ish?) into small functors that take
  "invariant" values (the larger context) for constructors and use
  ``operator()`` to transform some input into the desired output
- Use only ``const`` data when sharing. Non-const shared data is almost like
  using global variables.
- Use ``OpaqueId`` instead of integers and magic sentinel values for
  integer identifiers that aren't supposed to be arithmetical.

Examples:

- Random number sampling: write a unit sphere sampling functor instead of
  replicating a polar-to-cartesian transform in a thousand places
- Cell IDs: Opaque IDs add type safety so that you can't accidentally convert a
  cell identifier into a double or switch a cell and material ID. Also makes
  code more readable of course.


Maximize code reuse
-------------------

No explanation needed.


Minimize compile time
---------------------

Code performance is important, but so is developer time. When possible,
minimize the amount of code touched by NVCC. (NVCC's error output is also
rudimentary compared to modern clang/gcc, so that's another reason to prefer
them compiling your code.)

Prefer single-state classes
---------------------------

As much as possible, make classes "complete" and valid after calling the
constructor. Don't have a series of functions that have to be called in a
specific order to put the class in a workable state.

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
offers good suggestions.

.. _ISO C++ guidelines: http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
.. _Google style guide: https://google.github.io/styleguide/cppguide.html

Style guidelines
================

Having a consistent code style makes it more readable and maintainable. (For
example, you don't have to guess whether a symbol is a function or class.)

Formatting
----------

Formatting is determined by the clang-format file inside the top-level
directory. One key restriction is the 80-column limit, which enables multiple
code windows to be open side-by-side. Generally, statements longer than 80
columns should be broken into sub-expressions for improved readability anyway.

Symbol names
------------

Functions should be verbs; classes should be names. As in standard Python
(PEP-8-compliant) code, classes should use ``CapWordsStyle`` and variables
``snake_case_style``.

Functors (classes whose instances act like a function) should be an *agent
noun*: the noun form of an action verb. Instances of a functor should be a
verb. For example::

   ModelEvaluator evaluate_something(parameters...);
   auto result = evaluate_something(arguments...);


File names
----------

All ``__device__`` and ``__global__`` code must be compiled with NVCC to generate
device objects. However, code that merely uses CUDA API calls such as
``cudaMalloc`` does *not* have to be compiled with NVCC. Instead, it only has to
be linked against the CUDA runtime library and include ``cuda_runtime_api.h``.
The exception to this is VecGeom's code, which compiles differently when run
through NVCC. (Macro magic puts much of the code in a different namespace.)

Since NVCC is slower and other compilers' warning/error output is more
readable, it's preferable to use NVCC for as little compilation as possible.
Furthermore, not requiring NVCC lets us play nicer with downstream libraries
and front-end apps. Host code will not be restricted to the minimum version
supported by NVCC (C++14).

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

Finally, we choose the convention of ``.cc`` for C++ translation units and
corresponding ``.hh`` files for C++ headers.

Thus we have the following rules:

- ``.hh`` is for C++ header code compatible with host compilers. The code in
  this file can be compatible with device code if it uses the
  ``CELER_FUNCTION`` family of macros defined in ``base/Macros.hh``.
- ``.cc`` is for C++ code that will invariably be compiled by the host
  compiler.
- ``.cu`` is for ``__global__`` kernels and functions that launch them,
  including code that initializes device memory.
- ``.cuh`` is for header files that require compilation by NVCC: contain
  ``__device __``-only code or include CUDA directives without ``#include
  <cuda_runtime_api.h>``.
- ``.cuda.hh`` and ``.cuda.cc`` require CUDA to be enabled and use CUDA runtime
  libraries and headers, but they do not execute any device code and thus can
  be built by a host compiler. If the CUDA-related code is guarded by ``#if
  CELERITAS_USE_CUDA`` macros then the special extension is not necessary. A
  ``.nocuda.cc`` file can specify alternative code paths to ``.cuda.cc`` files
  (mainly for error checking code).
- ``.mpi.cc`` and ``.nompi.cc`` code requires MPI to be enabled or disabled,
  respectively.

Some files have special extensions:
- ``.i.hh`` is for ``inline`` function implementations. If a function or member
  function is marked ``inline`` in the main header file, its
  definition should be provided here. No ``inline`` modifier is needed for the
  ``.i.hh`` definition but it *must* be present in the ``.hh`` file.
- ``.t.hh`` is for non-inlined ``template`` implementations: if they're marked
  ``inline`` in their corresponding declaration in the ``.hh``, their
  implementation should be provided here.
- ``.test.cc`` are unit test executables corresponding to the main ``.cc``
  file. These should only be in the main ``/test`` directory.


Variable names
--------------

Generally speaking, variables should have short lifetimes and should be
self-documenting. Avoid shorthand and "transliterated" mathematical
expressions: prefer ``constants::avogadro`` to ``N_A`` or express the constant
functionally with ``atoms_per_mole``.

Data management
===============

Data management should be isolated from data use as much as possible. This
allows low-level physics classes to operate on references to data using the
exact same device/host code. Furthermore, state data (one per track) and
shared data (definitions, persistent data, model data) should be separately
allocated and managed.

Params (model parameters)
  Provide a CPU-based interface to manage and provide access to constant shared
  GPU data, usually model parameters or the like. The store class itself  can
  only be accessed via host code, but it should use the
  ``celeritas::DeviceVector`` or ``thrust`` or ``VecGeom`` wrappers to manage
  on-device data. A store class can contain metadata (string
  names, etc.) suitable for host-side debug output and for helping related
  classes convert from user-friendly input (e.g. particle name) to
  device-friendly IDs (e.g. particle def ID).

State (state variables)
  Thread-local data that corresponds to the model/store. Usually a simple
  storage class, since the state is meaningless without the model parameters.

Views
  A standalone plain-old-data reference to data owned by a Model, a Store, or a
  State,
  used to transfer GPU pointers and other kernel parameters between host and
  device. Views can hold other views as data members. Inside unit tests for
  debugging, views can point to *host* data if the corresponding functions
  being called are also on-host.

Model/Access
  Device-compatible class that combines the state variables and model
  parameters. This can be named arbitrarily.

.. example::

   All SM physics particles share a common set of properties such as mass,
   charge; and each instance of particle has a particular set of
   associated variables such as kinetic energy. The shared data (SM parameters)
   reside in the ParticleStore, and the particle track properties are managed
   by a ParticleStates class.

   A separate class, the Particle, acts as an accessor to the stored data and
   can calculate properties that depend on both the state and parameters. For
   example, momentum depends on the mass of a particle (constant, set by the
   model), and speed depends on the particle's kinetic energy state.
