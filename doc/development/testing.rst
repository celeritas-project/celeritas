.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: none

.. _testing_and_debugging:

Testing and debugging
=====================

Each class must be thoroughly tested with an independent unit test in the
`test` directory.  For complete coverage, each function of the class should have
at least as many tests to cover all parts of code, and possibly as many as the
number of code flow paths. So, if your function has a single ``if`` statement, it
should have at least two tests (to make sure each branch is tested); and if it
has three ``if`` statements, it may need up to eight different tests to ensure
that all combinations are tested. (For further discussion, read about
`cyclomatic complexity`_.) It's useful in such cases to define helper
functions to better isolate conditionals from each other.

.. _cyclomatic complexity: https://en.wikipedia.org/wiki/Cyclomatic_complexity

Running CTest
-------------

When configured with ``CELERITAS_BUILD_TESTS`` (see :ref:`configuration`),
CTest_ will be automatically configured. Running through CTest sets special
environment variables for data, disabling GPUs, or testing the code itself.
CTest can be run either through ``ninja test`` or by manually invoking
``ctest``.  Two useful ways to run are ``ctest -V -R <test regex>``, which will
run one or more tests that match the given regular expression (such as
``corecel/math/``), and ``ctest -j --output-on-failure`` which runs in parallel
and prints only test failures.

.. _CTest: https://cmake.org/cmake/help/latest/manual/ctest.1.html

Using GoogleTest
----------------

`Google test`_ is very well documented, and because so much testing code exists
in AI training data, tools like ChatGPT and Copilot are very good at writing a
first pass at test code.
Celeritas defines a base class test harness with some utility functions:

.. doxygenclass:: celeritas::test::Test

as well as several macros that simplify testing with floating-point data (and
vectors thereof):

.. doxygendefine:: EXPECT_VEC_EQ
.. doxygendefine:: EXPECT_REAL_EQ
.. doxygendefine:: EXPECT_SOFT_EQ
.. doxygendefine:: EXPECT_SOFT_NEAR
.. doxygendefine:: EXPECT_VEC_SOFT_EQ
.. doxygendefine:: EXPECT_VEC_NEAR
.. doxygendefine:: EXPECT_JSON_EQ
.. doxygendefine:: PRINT_EXPECTED

For more details on the test harnesses, especially the hierarchy used for
setting up physics problems for testing, see the ``celeritas::test`` namespace
in the Doxygen developer documentation.

You can run most tests manually from the build directory and filter so that
only a subset of tests run::

   $ ./test/celeritas/global_Stepper --gtest_filter=SimpleComptonTest.host

.. _Google test: https://google.github.io/googletest/

Using LLDB
----------

LLVM's built-in debugger is a very helpful tool for understanding what may be
going wrong (or right!) in the code. It's best if you can reduce a bug to
the simplest form that will run in a unit test with a debug assertion failure.
Then you can run lldb, telling it to break on C++ exception throws and perform
a backtrace after running, while telling GoogleTest to filter on the failing
test::

   $ lldb -o "break set -E c++" -o "run" -o "bt" -- ./test/celeritas/optical_Cerenkov --gtest_filter=CerenkovTest.generator
   (lldb) target create "./test/celeritas/optical_Cerenkov"
   Current executable set to '/Users/seth/Code/celeritas/build/test/celeritas/optical_Cerenkov' (arm64).
   (lldb) settings set -- target.run-args  "--gtest_filter=CerenkovTest.generator"
   (lldb) break set -E c++
   Breakpoint 1: no locations (pending).
   (lldb) run
   2 locations added to breakpoint 1
   Celeritas version 0.5.0-dev.209+dc984b0d8
   Note: Google Test filter = CerenkovTest.generator
   [==========] Running 1 test from 1 test suite.
   [----------] Global test environment set-up.
   [----------] 1 test from CerenkovTest
   [ RUN      ] CerenkovTest.generator
   Process 67474 stopped
   * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
       frame #0: 0x182ef4158 libc++abi.dylib`__cxa_throw
   libc++abi.dylib`:
   ->  0x182ef4158 <+0>:  pacibsp
       0x182ef415c <+4>:  stp    x22, x21, [sp, #-0x30]!
       0x182ef4160 <+8>:  stp    x20, x19, [sp, #0x10]
       0x182ef4164 <+12>: stp    x29, x30, [sp, #0x20]
   Target 0: (optical_Cerenkov) stopped.
   Process 67474 launched: '/Users/seth/Code/celeritas/build/test/celeritas/optical_Cerenkov' (arm64)
   (lldb) bt
   * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
     * frame #0: 0x182ef4158 libc++abi.dylib`__cxa_throw
       frame #1: 0x100017f98 optical_Cerenkov`celeritas::RejectionSampler<double>::RejectionSampler(this=0x16fdfcda8, f=-0.0062093880005715963, fmax=0.17188544207007173) at RejectionSampler.hh:87:5
       frame #2: 0x10001714c optical_Cerenkov`celeritas::RejectionSampler<double>::RejectionSampler(this=0x16fdfcda8, f=-0.0062093880005715963, fmax=0.17188544207007173) at RejectionSampler.hh:86:1
       frame #3: 0x100014c64 optical_Cerenkov`celeritas::Span<celeritas::optical::Primary, 18446744073709551615ul> celeritas::optical::CerenkovGenerator::operator()<celeritas::test::DiagnosticRngEngine<std::__1::mersenne_twister_engine<unsigned int, 32ul, 624ul, 397ul, 31ul, 2567483615u, 11ul, 4294967295u, 7ul, 2636928640u, 15ul, 4022730752u, 18ul, 1812433253u>>>(this=0x16fdfd1f8, rng=0x16fdfdc48) at CerenkovGenerator.hh:165:18
       frame #4: 0x10000ed60 optical_Cerenkov`celeritas::test::CerenkovTest_generator_Test::TestBody()::$_0::operator()(this=0x16fdfdb20, pre_step=0x16fdfd910, particle=0x16fdfd8d0, sim=0x16fdfd8a8, pos=0x16fdfd890, num_samples=64) const at Cerenkov.test.cc:361:28
       --8<-- snip --8<--

Many classes in Celeritas store complex structures of data. Normally LLDB does
not understand the various data pointers, so "collection groups" (such as
Params data) are unintelligible::

   (lldb) print params->host_ref()
   (const celeritas::ParamsDataInterface<celeritas::optical::CerenkovData>::HostRef) {
     angle_integral = {
       storage_ = {
         data = {
           s_ = {
             data = 0x600001e2faa0
             size = 1
           }
         }
       }
     }
     reals = {
       storage_ = {
         data = {
           s_ = {
             data = 0x00014282ac00
             size = 202
           }
         }
       }
     }
   }

You can execute these commands (note that this assumes the working
directory is one below the source, as it would if running in ``build``)::

   command script import ../scripts/dev/celerlldb.py --allow-reload
   type synthetic add -x "^celeritas::Span<.+>$" --python-class celerlldb.SpanSynthetic
   type synthetic add -x "^celeritas::ItemRange<.+>$" --python-class celerlldb.ItemRangeSynthetic

Then the "spans" of data will print their actual contents::

   (lldb) print params->host_ref()
   (const celeritas::ParamsDataInterface<celeritas::optical::CerenkovData>::HostRef) {
     angle_integral = {
       storage_ = {
         data = {
           [0] = {
             grid = (begin = 0, end = 0)
             value = (begin = 0, end = 0)
           }
         }
       }
     }
     reals = {
       storage_ = {
         data = {
           [0] = 0.0000010981771340407463
           [1] = 0.0000011070017717250021
           [2] = 0.0000011169747606594615
       --8<-- snip --8<--

For large data structures , you can prevent LLDB from eliding the
deep/long data::

   set set target.max-children-depth 16
   set set target.max-children-count 1024

When trying to debug a failure on CPU in the main Celeritas stepping loop, you
can call a global function to print the full state of the current track::

   (lldb) call celeritas::debug_print(track)
   {
    "geo": {
     "dir": [
      0.9998302826766889,
      0.010529089939196719,
      0.015117675340624488
     ],
     "is_on_boundary": false,
     "is_outside": false,
     "pos": [
      -2.135075225174846,
      0.0,
      0.0
     ],
     "volume_id": "inner@0x60000350ada0"
    },
    ...

If the stepping loop "hangs" (i.e., the number of steps seems unbounded) and
you have access to a debugger, you can call the ``Stepper::kill_active`` method
to kill all active tracks and (on CPU) print detailed debug information about
them.

.. _debug_print: https://github.com/celeritas-project/celeritas/pull/1304
