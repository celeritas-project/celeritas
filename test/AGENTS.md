# Testing

Googletest-based unit and regression testing.

## C++ code

### Test harness

- `@test/Test.hh`: global test harness
- FIXME: test base classes

### Macros

See `@test/TestMacros.hh`

- Floating point values: `EXPECT_SOFT_EQ`, `EXPECT_VEC_SOFT_EQ`
- Reference structs: `EXPECT_REF_EQ`
- JSON strings: `EXPECT_JSON_EQ`

Use `PRINT_EXPECTED` to dump reference values when updating expected results.

## CMake integration

Tests are GoogleTest executables registered in `test/**/CMakeLists.txt` via
`celeritas_add_test(Foo.test.cc)` or `celeritas_add_device_test(Foo)` (which
adds `Foo.test.cu` when CUDA/HIP is enabled).
