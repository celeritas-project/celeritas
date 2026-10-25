---
name: reference-result
description: 'Define or modify printable Celeritas test reference-result structs and IsRefEq comparators for EXPECT_REF_EQ or EXPECT_REF_NEAR. Use for reference data, tolerance structs, copy-pastable expected values, and field-level assertion diagnostics.'
argument-hint: 'Describe the result type, its fields, and whether comparisons need a tolerance'
---

# Celeritas Reference Results

Create or update a test-only reference-result data struct, its initializer
stream operator, `print_expected()` helper, and `IsRefEq` overloads. Reference
results commonly store vectors of POD values collected from a simulation or
geometry operation.

## 1. Find the local pattern

Inspect the owning test helper and nearby call sites before editing:

- `test/AssertionHelper.hh` for accumulating multiple failure messages.
- `test/testdetail/TestMacrosImpl.hh` and `test/TestMacros.hh` for
  `EXPECT_REF_EQ` / `EXPECT_REF_NEAR` and container dispatch.
- Existing result types with `IsRefEq`, `print_expected()`, or stream
  operators, especially `test/geocel/GenericGeoResults.*`,
  `test/larceler/RunnerResults.*`, and
  `test/celeritas/field/FieldPropagator.test.cc`.

Keep the result in the existing test helper or test file that owns the data;
follow its namespace, declaration/definition split, and CMake registration
patterns. For new C++ files, follow the `new-class` skill.

## 2. Define the result and optional tolerance

- Keep stored data simple and printable. Prefer vectors of POD or other
  already-printable values; use existing ID, units, and representation types
  where appropriate.
- Add a separate tolerance struct only when fields need approximate or fuzzy
  comparison. Give it one clearly named tolerance per independently
  configurable comparison, or reuse one tolerance when semantics are truly
  shared.
- Keep exact fields exact even when another field uses a tolerance.
- Preserve existing field order and public API when modifying a result type.

## 3. Stream a C++17 initializer

Implement `operator<<` to write a valid, copy-pastable initializer for the
result type, in declaration order. Celeritas is C++17: do not use C++20
designated initializers. Reuse `repr` or existing stream support for member
values, including nested vectors, rather than inventing formatting for them.
Return the supplied stream.

The output should be a brace initializer, for example:

```cpp
{{1, 2}, {3.0, 4.0}}
```

Make `print_expected() const` use this operator to print a complete test
snippet that declares the reference value, invokes the appropriate assertion,
and can be pasted into the owning test. For example, emit `FooResult ref = `,
then `*this`, then the assertion line. Include a tolerance declaration and use
`EXPECT_REF_NEAR` when the result has a tolerance type. Keep any special
handling for omitted, sentinel, or non-deterministic fields explicit in the
generated snippet.

## 4. Implement `IsRefEq`

Provide the overloads expected by the existing macros:

- Exact comparison: `IsRefEq(expected_expr, actual_expr, expected, actual)`.
- Tolerance comparison when needed:
  `IsRefEq(expected_expr, actual_expr, tol_expr, expected, actual, tol)`.
- If a tolerance type exists, also provide a no-tolerance overload for
  `EXPECT_REF_EQ`; delegate it to the tolerance overload with a
  default-initialized tolerance so the default comparison is exact.

Use `celeritas::test::AssertionHelper` to accumulate diagnostics instead of
returning on the first mismatch. Compare every field and add a field label to
each failure. For vectors, prefer the existing `testdetail::IsVecEq` and
`IsVecSoftEquiv` helpers where their semantics fit. Otherwise, check sizes
before indexing or iterating paired elements, then report element-level
mismatches without reading past either vector. Use exact comparison for
discrete fields and the appropriate local soft/fuzzy comparison for tolerant
fields. Include the tolerance expression and value in diagnostics when that
helps explain a failure.

Return the `AssertionHelper` so callers receive all accumulated messages. Do
not silence mismatches or compare unrelated fields under one broad tolerance.

### Preserve soft vector comparisons

When replacing an `EXPECT_VEC_SOFT_EQ` assertion with a result-struct
comparison, call `::celeritas::testdetail::IsVecSoftEquiv` for that field.
This uses the same default soft tolerances as the original macro. Follow the
local pattern in `test/corecel/random/HistogramSampler.cc` and
`test/larceler/RunnerResults.cc`: check field sizes with
`AssertionHelper::equal_size`, prefix helper failures with the field name, and
report the actual field value when the sizes differ. For example:

```cpp
if (result.equal_size(val1.values.size(), val2.values.size()))
{
  auto values_result = ::celeritas::testdetail::IsVecSoftEquiv(
    expr1, expr2, val1.values, val2.values);
  if (!values_result)
  {
    result.fail() << "  values: " << values_result.message();
  }
}
else
{
  result.fail() << "  values: " << expr2 << " = " << repr(val2.values);
}
```

Repeat this independently for each vector field so diagnostics identify which
field failed and the comparator continues collecting other field failures.

## 5. Add or update focused tests

Exercise both the comparator and the generated reference snippet workflow:

- Equal values pass exact and, where applicable, tolerance comparisons.
- A mismatch in each field fails and identifies that field.
- Different vector sizes fail cleanly without out-of-bounds access.
- Tolerance boundary behavior matches the existing comparison helper
  semantics.
- `operator<<` output is a valid initializer for the result type, and
  `print_expected()` includes the expected assertion form.
- If the type is used through `EXPECT_REF_EQ` / `EXPECT_REF_NEAR` with
  containers, verify the existing container-dispatch behavior still applies.

Build and run the narrow affected tests using the `test-change` skill. Report
any required build configuration that prevents those tests from running.
