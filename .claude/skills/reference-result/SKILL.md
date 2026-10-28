---
name: reference-result
description: 'Define or modify printable Celeritas test reference-result structs and is_ref_eq/check_ref_eq customizations for EXPECT_REF_EQ or EXPECT_REF_NEAR. Use for reference data, tolerance structs, copy-pastable expected values, and field-level assertion diagnostics.'
argument-hint: 'Describe the result type, its fields, and whether comparisons need a tolerance'
---

# Celeritas Reference Results

Create or improve a test-only reference-result struct, its reusable comparison
logic, and expected-value output. Reference results commonly store vectors of
values collected from a simulation or geometry operation. Implement detailed
type-specific comparison policy in `is_ref_eq` where it can be reused by other
reference results; keep `check_ref_eq` as the ADL customization that dispatches
`EXPECT_REF_EQ` and `EXPECT_REF_NEAR`. Follow the implemented interfaces in the
checked-in tree; distinguish existing behavior from proposed extensions.

## 1. Find the local pattern

Inspect the owning test helper and nearby call sites before editing:

- `test/AssertionHelper.hh` for accumulating field-level failure messages.
- `test/testdetail/TestMacrosImpl.hh` for the central `IsRefEq` predicate
  dispatchers, generic container `check_ref_eq`, and field-aware helpers such
  as `is_eq`, `is_vec_eq`, and `is_vec_soft_eq`; `test/TestMacros.hh` for
  `EXPECT_REF_EQ` / `EXPECT_REF_NEAR`.
- `test/accel/HepMC3PrimaryGenerator.test.cc` as a canonical example.

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
- Checks of individual fields can be omitted deliberately by clearing both
  reference and actual results.
- Add a separate tolerance struct only when fields need approximate or fuzzy
  comparison. Give it one clearly named tolerance per independently
  configurable comparison, or reuse one tolerance when semantics are truly
  shared.
- When a tolerance struct supplies field-level defaults, use the matching
  tolerance member for every soft-compared field, including fields using the
  default value. Avoid boolean flags that switch between default and custom
  comparisons. Set wider per-test tolerances, such as `coarse_eps`, at the test
  call site; keep discrete fields exact.
- Keep exact fields exact even when another field uses a tolerance.
- Preserve existing field order and public API when modifying a result type.
- At assertion call sites, put expected values directly in the reference
  object, using an aggregate initializer or assignments to its fields. Avoid
  temporary or static `expected_*` arrays/vectors used only to initialize the
  reference. For version-dependent values, initialize the reference with the
  common values and assign the differing field directly, such as `ref.xs`.

For example:

```cpp
ImportXsSummary ref{{0, 0}, {0.0001, 100000000, 0.0001, 100000000}, {}};
EXPECT_REF_EQ(ref, result) << result;
```

or:
```cpp
ImportXsSummary ref;
ref.size = {0, 0};
ref.energy = {0.0001, 100000000, 0.0001, 100000000};
EXPECT_REF_EQ(ref, result) << result;
  ```

## 3. Print expected values

There is no requirement to add an initializer stream operator. Choose the
smallest output form that is copy-pastable and fits the result type:

- For field-based structs, use `CELER_REF_ATTR(field)` and `repr` in
  `print_expected()` or an existing stream helper. This is the common pattern
  when a result has no useful aggregate initializer.
- Add `operator<<` when an aggregate initializer is useful or when the existing
  test convention streams a complete expected-value snippet. Return the
  supplied stream.
- When streaming an aggregate result, have `operator<<` emit only a valid
  aggregate initializer in declaration order. Keep snippet framing markers and
  the assertion in `print_expected()`, which can dispatch with `*this` to the
  stream operator. Keep `print_expected()` parameterless and use the default
  comparison (`EXPECT_REF_EQ`) there; per-test tolerance overrides belong at
  the assertion call site.
- Keep handling for omitted, sentinel, or non-deterministic fields explicit in
  the generated snippet, with a short explanation when needed.

If an aggregate initializer is chosen, it must be valid C++17, in declaration
order, and reuse `repr` or existing stream support for member values. Do not
use C++20 designated initializers. `print_expected()` should emit a complete
snippet for the owning test, including the right assertion macro and any
required tolerance setup. Test the emitted text against the expected output.

## 4. Implement reusable comparison logic and dispatch

`EXPECT_REF_EQ` and `EXPECT_REF_NEAR` call the central predicate-format
dispatchers in `celeritas::testdetail`; do not add result-specific `IsRefEq`
overloads. The dispatcher creates one `celeritas::test::AssertionHelper` and
calls a type-specific `check_ref_eq` customization. Keep this customization as
a thin `void` adapter that delegates to the reusable `bool is_ref_eq` policy:

```cpp
bool is_ref_eq(AssertionHelper& helper,
         char const* field_name,
         Result const& expected,
         Result const& actual,
         std::nullptr_t);

bool is_ref_eq(AssertionHelper& helper,
         char const* field_name,
         Result const& expected,
         Result const& actual,
         Tol const& tol);

void check_ref_eq(AssertionHelper& helper,
          Result const& expected,
          Result const& actual,
          std::nullptr_t)
{
  (void)is_ref_eq(helper, "Result", expected, actual, nullptr);
}

void check_ref_eq(AssertionHelper& helper,
          Result const& expected,
          Result const& actual,
          Tol const& tol)
{
  (void)is_ref_eq(helper, "Result", expected, actual, tol);
}
```

Implement the matching `is_ref_eq` policy overloads for exact comparison and
for any supported tolerance. The `std::nullptr_t` overload corresponds to
`EXPECT_REF_EQ`; the tolerance overload corresponds to `EXPECT_REF_NEAR`. Keep
the `check_ref_eq` adapter overloads in the result's associated namespace so
ADL can find them. The no-tolerance tag is passed explicitly by the dispatcher;
do not substitute `{}` for a generic tolerance argument. Give `is_ref_eq` a
field name so a reusable comparison can produce useful diagnostics when called
from another result's comparator.

Define the customization in a namespace associated with the result type, or as
a hidden friend when needed for a nested/private result type, so argument-
dependent lookup can find it. Put non-inline declarations in the owning test
helper header and definitions in its source file. Include `AssertionHelper.hh`
where a complete helper type is needed; include `<cstddef>` for
`std::nullptr_t` declarations.

The helper already contains the expected and actual expression strings. For
`EXPECT_REF_NEAR`, `helper.tol_expr()` retains the tolerance expression for
field diagnostics. Customizations should append failures to the shared helper,
not construct or return a separate assertion result. Prefer the field-aware
wrappers from `test/testdetail/TestMacrosImpl.hh`:

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
helper by reference and let the central dispatcher return the accumulated
assertion result.

For library-owned template types, put their test-only `is_ref_eq` policy and
`check_ref_eq` adapter in the type's associated test namespace. Other reference
result comparators can call `is_ref_eq` directly for a nested field, passing
the field name and the desired tolerance; this keeps the detailed policy in
one place. If only selected template instantiations are supported, declare and
explicitly instantiate the policy as appropriate. Exercise direct
`EXPECT_REF_EQ` as well as `EXPECT_REF_NEAR`, including tolerance boundaries
and any special values or null-state behavior.

The generic container customization is provided centrally. It checks size,
recurses through the central dispatcher for each element, labels failures with
the element index, and caps the number of printed failures. Do not write a
result-specific container overload or recurse through an unqualified `IsRefEq`
call, which could select an older ADL predicate instead of the central
dispatcher. Add a focused container test when a result type is expected to be
used inside containers.

### Preserve soft vector semantics

When replacing an `EXPECT_VEC_SOFT_EQ` assertion with a result-struct
comparison, use `is_vec_soft_eq` for that field. It preserves the macro's
default soft tolerances while checking sizes, labeling diagnostics, and
continuing to later fields. `is_vec_eq` provides the corresponding exact
container behavior. Follow the current `RunnerResults.cc` and
`HistogramSampler.cc` patterns rather than duplicating these wrapper checks.
When a tolerance struct is used, always use the five-argument
`is_vec_soft_eq` overload for each soft-compared field, passing that field's
tolerance member. This keeps size checks, field labels, and failure
accumulation consistent while making default and overridden tolerances
explicit.

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
