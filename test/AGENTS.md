# Test Instructions

Tests are C++ GoogleTest executables registered with CTest. Keep tests focused
on observable behavior, and cover normal cases, boundary conditions, and error
paths that are part of the contract. When changing a class or function, extend
its existing test before creating a new test file.

## File placement and layout

- Mirror the `src/` directory beneath `test/`. Put `.test.cc` files in the
  corresponding `test/<library>/` directory and register them in that
  directory's `CMakeLists.txt`.
- Name a test file for the class or closely related functionality it covers.
  A test may cover multiple related classes; add a `\sa test/...` reference to
  the source file's Doxygen file comment when the name does not match the class.
- Include the production header, then the narrowest required test helpers. Most
  tests include `celeritas_test.hh`; it exports the harness, test macros, and
  test `main`. Include `TestMacros.hh` directly only when the test does not
  need that meta-include.
- Put tests in `celeritas::test`. For implementation details in
  `celeritas::detail`, use `celeritas::detail::test` and refer to
  `celeritas::test::Test` explicitly.
- Use `TEST` for independent cases. Use a fixture derived from
  `celeritas::test::Test` with `TEST_F` when cases share setup or need harness
  helpers. Avoid shared mutable state between tests.

## Harness, assertions, and expected values

- `@test/Test.hh` defines the common fixture. Use `test_data_path(subdir,
  filename)` for checked-in data under `test/<subdir>/data/`, and
  `make_unique_filename()` for temporary output so parallel tests do not
  collide. Use the `is_reference_*` flags when an expected result is only
  valid for the reference RNG, geometry, or physics configuration.
- `@test/TestMacros.hh` defines Celeritas comparisons. Use `EXPECT_SOFT_EQ`
  and `EXPECT_VEC_SOFT_EQ` for approximate floating-point results;
  `EXPECT_SOFT_NEAR` and `EXPECT_VEC_NEAR` take an explicit relative tolerance.
  Use `EXPECT_REAL_EQ` for single-ULP equality, `EXPECT_VEC_EQ` for exact
  containers, `EXPECT_REF_EQ` for reference-comparable structs, and
  `EXPECT_JSON_EQ` for JSON values/strings. Prefer ordinary GoogleTest
  assertions for exact integral, enum, boolean, and string results.
- Keep expected values independent of the implementation under test. For
  stochastic behavior, use deterministic test inputs/seeds and verify stable
  properties or distributions rather than relying on an uncontrolled random
  sequence.
- When changing regression data, use `PRINT_EXPECTED` to inspect the produced
  values, then review and update the expected values deliberately. Do not
  replace meaningful expectations with snapshots of the current output without
  checking the behavioral change.
- `CELER_EXPECT`, `CELER_ASSERT`, and `CELER_ENSURE` checks are configuration-
  dependent. Test debug-only failures under `CELERITAS_DEBUG`; use the
  `TEST_IF_CELERITAS_DEBUG`, `TEST_IF_CELER_DEVICE`,
  `TEST_IF_CELERITAS_DOUBLE`, `TEST_IF_CELERITAS_GEANT`, or
  `TEST_IF_CELERITAS_USE_ROOT` macros when a GoogleTest case only applies to
  that configuration. Do not disable a test merely to avoid fixing a failure.

## Host and device tests

- Use a regular `.test.cc` for host behavior. For behavior that must compile
  and execute on CUDA/HIP, pair `<Name>.test.cc` with `<Name>.test.cu` and
  `<Name>.test.hh` when shared declarations are needed. Keep kernels and device
  launches in `.test.cu`; keep host-side test cases and assertions in
  `.test.cc`.
- Register the pair with `celeritas_add_device_test(<Name>)`. This adds the
  `.test.cu` source and GPU resource lock when CUDA or HIP is enabled, while
  retaining the host test in non-device builds. Compare device results against
  explicit expected values or a host result where appropriate.

## CMake registration

- Add each test to the owning `test/**/CMakeLists.txt` using
  `celeritas_add_test(<path>/<Name>.test.cc)`. Use
  `celeritas_add_device_test(<path>/<Name>)` for a `.test.cc`/`.test.cu` pair.
- Add only required options: `LINK_LIBRARIES` for extra link targets,
  `SOURCES` for additional compiled sources, `ENVIRONMENT` for required test
  variables, `ADD_DEPENDENCIES` for generated/build prerequisites, and
  `GPU` to serialize tests that use a GPU without a device kernel.
- `celeritas_add_test` also supports `FILTER` for splitting a large executable
  into independent CTest cases, `NP` for MPI process counts, `TIMEOUT`,
  `DEPTEST`, and `DISABLE`. Use these only when the test needs that behavior;
  `DISABLE` omits execution from CTest but still builds the executable, so
  document why it is necessary.
- Do not add a custom `main` or call `add_test` directly for ordinary C++
  tests. The shared harness and CMake helpers configure GoogleTest, CTest,
  MPI variants, and test environment consistently.

## Running and debugging

Configure with `CELERITAS_BUILD_TESTS` enabled, build the test target, then run
the smallest relevant CTest selection from the build directory. For example:

```sh
ctest --test-dir build -R 'corecel/math/SpanUtils' --output-on-failure
ctest --test-dir build -V -R 'corecel/math/'
```

CTest's registered name may differ from the executable path, especially for
MPI or filtered tests; use `ctest -N` to inspect matching names. For a quick
iteration, a test executable can also be run directly with
`--gtest_filter=SuiteName.TestName`. See `@doc/development/testing.rst` for the
full testing and debugging guide.
