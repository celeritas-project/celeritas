---
name: reference-result
description: 'Define or modify printable Celeritas test reference-result structs and IsRefEq comparators for EXPECT_REF_EQ or EXPECT_REF_NEAR. Use for reference data, tolerance structs, copy-pastable expected values, and field-level assertion diagnostics.'
argument-hint: 'Describe the result type, its fields, and whether comparisons need a tolerance'
---

# Celeritas Reference Results

Create or improve a test-only reference-result struct and its `IsRefEq`
comparison and expected-value output. Reference results commonly store vectors
of values collected from a simulation or geometry operation. Follow the
implemented interfaces in the checked-in tree; distinguish existing behavior
from proposed extensions.

## 1. Find the local pattern

Inspect the owning test helper and nearby call sites before editing:

- `test/AssertionHelper.hh` for accumulating field-level failure messages.
- `test/testdetail/TestMacrosImpl.hh` for `is_eq`, `is_vec_eq`,
  `is_vec_soft_eq`, scalar/vector comparison predicates, and generic container
  dispatch; `test/TestMacros.hh` for `EXPECT_REF_EQ` / `EXPECT_REF_NEAR`.
- Existing result types with `IsRefEq`, `print_expected()`, or stream
  operators, especially `test/geocel/GenericGeoResults.*`,
  `test/larceler/RunnerResults.*`, and
  `test/celeritas/field/FieldPropagator.test.cc`.

Keep the result in the existing test helper or test file that owns the data;
follow its namespace, declaration/definition split, and CMake registration
patterns. For new C++ files, follow the `new-class` skill.

## 2. Define or improve the result and optional tolerance

- Keep stored data simple and printable. Prefer vectors of POD or other
  already-printable values; use existing ID, units, and representation types
  where appropriate.
- When improving an existing result, inventory every field and compare that
  inventory against its construction, expected-value output, and comparator.
  Preserve intentional omissions, sentinels, and normalization behavior
  explicitly; do not silently drop fields or change comparison semantics.
- Add a separate tolerance struct only when fields need approximate or fuzzy
  comparison. Give it one clearly named tolerance per independently
  configurable comparison, or reuse one tolerance when semantics are truly
  shared.
- Keep exact fields exact even when another field uses a tolerance.
- Preserve existing field order and public API when modifying a result type.

## 3. Print expected values

There is no requirement to add an initializer stream operator. Choose the
smallest output form that is copy-pastable and fits the result type:

- For field-based structs, use `CELER_REF_ATTR(field)` and `repr` in
  `print_expected()` or an existing stream helper. This is the common pattern
  when a result has no useful aggregate initializer.
- Add `operator<<` when an aggregate initializer is useful or when the existing
  test convention streams a complete expected-value snippet. Return the
  supplied stream.
- Keep handling for omitted, sentinel, or non-deterministic fields explicit in
  the generated snippet, with a short explanation when needed.

If an aggregate initializer is chosen, it must be valid C++17, in declaration
order, and reuse `repr` or existing stream support for member values. Do not
use C++20 designated initializers. `print_expected()` should emit a complete
snippet for the owning test, including the right assertion macro and any
required tolerance setup. Test the emitted text against the expected output.

## 4. Implement `IsRefEq` with the current helpers

`EXPECT_REF_EQ` calls the four-argument `IsRefEq(expected_expr, actual_expr,
expected, actual)` overload. `EXPECT_REF_NEAR` calls the six-argument overload
`IsRefEq(expected_expr, actual_expr, tol_expr, expected, actual, tol)`. These
are predicate-format APIs found by ADL; preserve their signatures and namespace
placement. Provide both overloads when a tolerance type is needed. Define the
no-explicit-tolerance behavior deliberately: a default-initialized tolerance
does not necessarily mean exact comparison if the comparator gives it soft
defaults.

Start with `celeritas::test::AssertionHelper helper{expr1, expr2};` and prefer
the field-aware wrappers from `test/testdetail/TestMacrosImpl.hh`:

- `is_eq(helper, "field", expected, actual)` for exact scalar or directly
  comparable fields.
- `is_vec_eq(helper, "field", expected, actual)` for exact containers.
- `is_vec_soft_eq(helper, "field", expected, actual)` for default soft
  container equivalence, including its size check and field-labeled diagnostic.
- `is_vec_soft_eq(helper, "field", expected, actual, tol)` for a field-specific
  tolerance or comparator, with the same size check and field-labeled
  diagnostic. The final argument can be a scalar tolerance or a soft comparator
  such as `SoftEqual` or `EqualOr`.
- `is_soft_eq` with the appropriate comparator for scalar tolerant fields.

These wrappers append failures to the shared helper so comparison continues
across fields. Use `IsVecEq` or `IsVecSoftEquiv` directly only when their
specific behavior is needed that the field-aware wrappers cannot express; add
the field label and protect any manual paired traversal with a size check.
Compare discrete fields exactly, keep tolerances local to the fields they
govern, and include tolerance context in diagnostics when useful. Return the
helper so callers receive accumulated failures.

### Preserve soft vector semantics

When replacing an `EXPECT_VEC_SOFT_EQ` assertion with a result-struct
comparison, use `is_vec_soft_eq` for that field. It preserves the macro's
default soft tolerances while checking sizes, labeling diagnostics, and
continuing to later fields. `is_vec_eq` provides the corresponding exact
container behavior. Follow the current `RunnerResults.cc` and
`HistogramSampler.cc` patterns rather than duplicating these wrapper checks.
For custom tolerance behavior, use the five-argument `is_vec_soft_eq` overload
so size checks, field labels, and failure accumulation remain consistent.

## 5. Add or update focused tests

Exercise both the comparator and the generated reference snippet workflow:

- Equal values pass exact and, where applicable, tolerance comparisons.
- Mismatches fail and identify the affected fields; cover each special
  comparison policy.
- Different vector sizes fail cleanly without out-of-bounds access.
- Tolerance boundary behavior matches the existing comparison helper
  semantics.
- If `operator<<` is provided, its output is a valid initializer; verify that
  `print_expected()` includes the expected assertion and tolerance setup.
- If the type is used through `EXPECT_REF_EQ` / `EXPECT_REF_NEAR` with
  containers, verify the existing container-dispatch behavior still applies.
- For an existing struct, add focused regression coverage for corrected
  omissions, field policies, or diagnostics without changing unrelated
  semantics.

Build and run the narrow affected tests using the `test-change` skill. Report
any required build configuration that prevents those tests from running.

## 6. Plan capability extensions

Treat these as extension points, not current guarantees of the API. Before
implementing a requested capability, specify the supported field shape, exact
versus tolerant semantics, diagnostic requirements, generated snippet form,
and compatibility expectations. Verify the owning helper interface and add a
focused test before documenting the capability as available.

Potential gaps to evaluate when a real use case appears:

- Composition of nested result comparators without losing field paths in
  diagnostics.
- Explicit policies for optional, ignored, or sentinel-valued fields in both
  comparison and generated expected snippets.

Keep a proposal separate from implemented usage guidance, and avoid adding a
generic abstraction until multiple concrete result types need it.
