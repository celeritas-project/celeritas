---
name: reference-result
description: 'Define or modify printable Celeritas test reference-result structs and is_ref_eq/check_ref_eq customizations for EXPECT_REF_EQ or EXPECT_REF_NEAR. Use for reference data, tolerance structs, copy-pastable expected values, and field-level assertion diagnostics.'
argument-hint: 'Describe the result type, its fields, and whether comparisons need a tolerance'
---

# Celeritas Reference Results

Create or improve a test-only reference-result struct, its comparison logic,
and expected-value output. Reference results commonly store vectors of values
collected from a simulation or geometry operation. The usual case is a
test-specific result: define `check_ref_eq` as its ADL customization and put
Use field-by-field `check_ref_eq` only when a particular `Result` needs special handling, such as
ignored or conditionally compared fields. Define reusable `is_ref_eq` only
for a class the user asks to compare as part of the result; `BoundingBox` is
an example. Do not add a reusable policy just because a result struct exists.
Follow the checked-in interfaces and distinguish implemented behavior from
proposed extensions.

## 1. Find the local pattern

Inspect the owning test helper and nearby call sites before editing:

- `test/AssertionHelper.hh` for accumulating field-level failure messages.
- `test/testdetail/TestMacrosImpl.hh` for the central `IsRefEq` predicate
  dispatchers, generic container `check_ref_eq`, and field-aware helpers such
  as `is_eq`, `is_vec_eq`, and `is_vec_soft_eq`; `test/TestMacros.hh` for
  `EXPECT_REF_EQ` / `EXPECT_REF_NEAR`.
- `test/accel/HepMC3PrimaryGenerator.test.cc` as a canonical example.

Keep the result in the test helper or test file that owns the data. For new
C++ files, follow the `new-class` skill.

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
- Set wider per-test tolerances, such as `coarse_eps`, at the call site. Keep
  discrete fields exact even when other fields are tolerant.
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
  aggregate initializer in declaration order. Keep snippet framing markers
  and the assertion in parameterless `print_expected()`, which can dispatch
  with `*this` to the stream operator. Use `EXPECT_REF_EQ` there; per-test
  tolerance overrides belong at the assertion call site.
- Keep handling for omitted, sentinel, or non-deterministic fields explicit in
  the generated snippet, with a short explanation when needed.

Use C++17 aggregate syntax, `repr` or existing member stream support, and no
designated initializers. Test that the generated snippet is valid and includes
the assertion and any required setup.

dependent lookup can find it. Put non-inline declarations in the owning test
## 4. Implement comparison dispatch and customizations

`EXPECT_REF_EQ` and `EXPECT_REF_NEAR` use central dispatch in
`celeritas::testdetail`; do not add result-specific `IsRefEq` overloads. The
dispatcher creates an `AssertionHelper` and calls `check_ref_eq`, which must
return `void`. A reusable `is_ref_eq` policy returns `bool`.

### Reusable class comparison

Define `is_ref_eq` only for a class the user asks to compare as part of a
reference result. Keep it focused on that class's comparison; do not use it to
traverse the fields of an enclosing `Result`. For main library classes, define
the policy in `celeritas::test`. ADL may not find this nested namespace for a
main-namespace class, so callers in other tests can use
`using celeritas::test::is_ref_eq;` before calling it.

Implement comparison logic once. If only exact or only tolerant comparison is
needed, define only that policy overload. If both are needed, make the
no-tolerance overload dispatch to the tolerance overload with a default value:

```cpp
bool is_ref_eq(AssertionHelper& helper,
               char const* field_name,
               Result const& expected,
               Result const& actual,
               std::nullptr_t)
{
  return is_ref_eq(
      helper, field_name, expected, actual, SoftEqual<>{}.rel());
}
```

The `std::nullptr_t` form serves `EXPECT_REF_EQ`; the tolerance form serves
`EXPECT_REF_NEAR`. A `check_ref_eq` adapter, when required by the dispatcher,
is an ADL customization returning `void` and delegates to `is_ref_eq`:

```cpp
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

Put main-library `is_ref_eq` declarations and definitions in the corresponding
test helper header/source when shared by tests. Explicitly instantiate only
the supported template types. Put `check_ref_eq` where the dispatcher can
find it by ADL, following the local pattern.

### Result-specific field comparison

Field-by-field `check_ref_eq` is a targeted, unusual customization for a
`Result` whose fields need special semantics, such as a skipped or
conditionally compared field, or independent tolerances. It is not part of a
reusable class's `is_ref_eq` policy. In this customization, append failures to
the shared helper and return `void`; do not return the helper or construct a
separate assertion result. The helper retains the expected, actual, and
tolerance expressions for diagnostics.

Implement only the comparison mode(s) the result uses. If both exact and
tolerance comparisons are needed, have the no-tolerance `check_ref_eq` forward
to the tolerance overload with its default tolerance (for example, `Tol{}`)
rather than duplicating field checks. Prefer these field-aware helpers:

- `is_eq(helper, "field", expected, actual)` for exact values.
- `is_vec_eq(helper, "field", expected, actual)` for exact containers.
- `is_vec_soft_eq(helper, "field", expected, actual)` for default soft vector
  equivalence.
- `is_vec_soft_eq(helper, "field", expected, actual, tol)` for a field's
  tolerance or comparator. When using a tolerance struct, pass the matching
  member for every soft-compared field.
- `is_soft_eq` with the appropriate comparator for scalar tolerant fields.

These helpers check vector sizes, label failures, and continue comparing later
fields. Keep discrete fields exact and tolerances local to their fields. Use
`IsVecEq` or `IsVecSoftEquiv` directly only when the wrappers cannot express
the needed behavior; label the field and check sizes before paired traversal.

For a shared customization, declare it in the owning test helper header and
define it in the corresponding source file. A test-local customization can
follow the existing translation-unit pattern. Include `AssertionHelper.hh`
where a complete type is needed and `<cstddef>` for `std::nullptr_t`.

The generic container `check_ref_eq` is provided centrally. It checks size,
dispatches each element, labels failures by index, and limits printed
diagnostics. Add a focused container test only when the result is used through
containers.

## 5. Add or update focused tests

Test each comparison policy and output path that the result uses:

- Equal values pass and mismatches fail with useful diagnostics; cover each
  custom comparison policy.
- Different vector sizes fail cleanly when the result contains vectors.
- Check tolerance boundaries when tolerant comparison is supported.
- If `operator<<` is provided, its output is a valid initializer; verify that
  `print_expected()` emits a usable snippet.
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
