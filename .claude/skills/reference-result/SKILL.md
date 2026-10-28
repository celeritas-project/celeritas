---
name: reference-result
description: 'Identify, create, and use printable Celeritas test reference-result structs and comparison helpers for EXPECT_REF_EQ or EXPECT_REF_NEAR. Use when inferring a result from repeated assertions, adding tolerances, refactoring comparisons, or printing expected values.'
argument-hint: 'Describe the test/file, result fields, and exact or tolerant comparison needs'
---

# Celeritas Reference Results

Create or improve a test-only reference-result struct, its comparison logic,
and expected-value output. Every reference-result class defines ADL-visible
friend functions for `print_expected`, `operator<<`, and `check_ref_eq`.
`print_expected` emits a copy-pastable expected result and assertion;
`operator<<` provides its result representation; `check_ref_eq` implements or
delegates comparison. Reusable reference-result classes also define
`is_ref_eq`, with `check_ref_eq` delegating to that policy. Field-by-field
`check_ref_eq` is appropriate when fields have special semantics, such as
independent tolerances or skipped fields. Follow the checked-in interfaces
and distinguish implemented behavior from proposed extensions.

Examples of the end result:

```cpp
auto result = run_test(1);
ImportXsSummary ref{{0, 0}, {0.0001, 100000000, 0.0001, 100000000}, {}};
EXPECT_REF_EQ(ref, result) << result;
// or
ImportXsSummary ref2;
ref2.size = {0, 0};
ref2.energy = {0.0001, 100000000, 0.0001, 100000000};
EXPECT_REF_EQ(ref2, result);

// To create a new reference result instance
auto new_result = run_test(2);
print_expected(new_result);
```

## 1. Identify the reference result struct

The reference type may be named by the user, already exist in the test, or be
inferred from a mostly repeated set of `EXPECT_...` checks in the user-provided
file. It may also be a simple library class from `src/` that tests should
compare directly, such as `BoundingBox`.

- Start from the provided file and nearby call sites. Identify the values
  repeatedly constructed or asserted, then inventory their fields and any
  intentional omissions, sentinels, or normalization behavior.
- Decide whether to reuse an existing type, create a test-only result type, or
- Decide whether to reuse an existing type, create a test-only result type, or
  compare a library class directly. Every reference-result class needs the
  friend functions described above; reusable result classes also need
  `is_ref_eq`.
- Determine which fields compare exactly and which need approximate or fuzzy
  comparison. Keep discrete values exact, and note per-test tolerance
  overrides needed at assertion sites.

Inspect the owning test helper and nearby call sites before editing:

- `test/AssertionHelper.hh` for accumulating field-level failure messages.
- `test/testdetail/TestMacrosImpl.hh` for the central `IsRefEq` predicate
  dispatchers, generic container `check_ref_eq`, and field-aware helpers such
  as `is_eq`, `is_vec_eq`, and `is_vec_soft_eq`; `test/TestMacros.hh` for
  `EXPECT_REF_EQ` / `EXPECT_REF_NEAR`.
- `test/accel/HepMC3PrimaryGenerator.test.cc` as a canonical example.

Keep the result in the test helper or test file that owns the data. For new
C++ files, follow the `new-class` skill.

## 2. Create missing reference, tolerance, and comparison helpers

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
- If an existing test compares only a small subset of a newly created
  reference class, it is acceptable to define the reference after getting
  the actual value using `auto ref = result;` and setting only the expected
  fields.

### Implement comparison dispatch and customizations

`EXPECT_REF_EQ` and `EXPECT_REF_NEAR` use central dispatch in
`celeritas::testdetail`; do not add result-specific `IsRefEq` overloads. The
dispatcher creates an `AssertionHelper` and calls the result's ADL-found
`check_ref_eq`, which must return `void`. A reusable `is_ref_eq` policy
returns `bool`. Define `check_ref_eq` as a friend of the result class so ADL
finds it.

### Reusable class comparison

Every reusable reference-result class defines `is_ref_eq` as a friend, keeping
the policy focused on that class rather than using it to traverse an enclosing
`Result`. A non-reusable, test-local result can compare its fields directly in
its friend `check_ref_eq`. For a main library class being compared as a
reusable result, define the policy in `celeritas::test`; ADL may not find this
nested namespace for a main-namespace class, so callers in other tests can use
`using celeritas::test::is_ref_eq;` before calling it.

Implement comparison logic once. If no custom tolerance or tolerance struct is
supported, define only one `is_ref_eq` overload, taking `std::nullptr_t`, and
only the matching `check_ref_eq` friend overload. Do not add tolerance
overloads. If a tolerance or tolerance struct is supported, define the
`is_ref_eq` tolerance overload and make its `std::nullptr_t` overload
dispatch to it with the appropriate default value:

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
`EXPECT_REF_NEAR`. Define `check_ref_eq` as a friend returning `void`; it
delegates to `is_ref_eq` for reusable result classes:

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

For main-library classes, put `is_ref_eq` declarations and definitions in the
corresponding test helper header/source when shared by tests. Explicitly
instantiate only the supported template types. Declare `check_ref_eq` as a
friend so the dispatcher can find it by ADL, following the local pattern.

### Result-specific field comparison

For a non-reusable result, or when fields need special semantics such as a
skipped or conditionally compared field or independent tolerances, implement
the friend `check_ref_eq` field by field. This is not a reusable class's
`is_ref_eq` policy. Append failures to the shared helper and return `void`; do
not return the helper or construct a separate assertion result. The helper
retains the expected, actual, and tolerance expressions for diagnostics.

If the result has no custom tolerance, define only the `check_ref_eq` overload
taking `std::nullptr_t`. If it supports a tolerance, define the `Tol` overload
and have the `std::nullptr_t` overload forward to it with the default
tolerance (for example, `Tol{}`) rather than duplicating field checks. Prefer
these field-aware helpers:

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

For a small test-local result declared at namespace scope, define its
`print_expected`, `operator<<`, and `check_ref_eq` as inline friends so ADL
finds them; define `is_ref_eq` as an inline friend too when the result is
reusable. Friend function definitions cannot be placed in a local class. For
substantial or shared customizations, declare the functions as friends in the
owning result class and define them in the corresponding test helper source.
Include `AssertionHelper.hh` where a complete type is needed and `<cstddef>`
for `std::nullptr_t`.

The generic container `check_ref_eq` is provided centrally. It checks size,
dispatches each element, labels failures by index, and limits printed
diagnostics. Add a focused container test only when the result is used through
containers.

### Compile checkpoint

After creating the reference type and comparison helpers, build the owning
test target and confirm that this scaffolding compiles. Use the local
configured build and the `test-change` skill to identify the narrow target.
If compilation fails, fix only this part and rebuild. Do not start Part 3
until compilation succeeds. If the target cannot be built because of missing
configuration or dependencies, report the blocker and stop before refactoring
assertion call sites.

## 3. Refactor existing comparisons or print expected values

Once the compile checkpoint passes, update existing checks to use the result
and its comparison helper, or print a copy-pastable expected result from the
test. Keep the change local to the identified test behavior.

### Refactor assertion call sites

- At assertion call sites, put expected values directly in the reference
  object, using an aggregate initializer or assignments to its fields. Avoid
  temporary or static `expected_*` arrays/vectors used only to initialize the
  reference.
- For version-dependent values, initialize the reference with common values
  and assign the differing field directly, such as `ref.xs`.
- If an existing test compares only a small subset of a newly created
  reference class, it is acceptable to define the reference after getting the
  actual value using `auto ref = result;` and setting only expected fields.
- Set wider per-test tolerances, such as `coarse_eps`, at the call site. Keep
  discrete fields exact even when other fields are tolerant.

For example:

```cpp
ImportXsSummary ref{{0, 0}, {0.0001, 100000000, 0.0001, 100000000}, {}};
EXPECT_REF_EQ(ref, result) << result;
```

### Print expected values

Every reference result defines friend `operator<<` and friend
`print_expected(Result const&)`. The stream operator provides the result
representation; `print_expected` uses it to emit a copy-pastable initializer,
assertion, and any required setup. Choose an output form that fits the result
type:

- For field-based structs, use `CELER_REF_ATTR(field)` and `repr` in the
  friend stream operator. This is the common pattern when a result has no
  useful aggregate initializer.
- When streaming an aggregate result, have `operator<<` emit only a valid
  aggregate initializer in declaration order. Keep snippet framing markers
  and the assertion in friend `print_expected(Result const&)`, which streams
  the result. Use `EXPECT_REF_EQ` there; per-test tolerance overrides belong
  at the assertion call site.
- For a small test-local result declared at namespace scope, define both
  functions inline as friends. Put substantial or shared output logic in the
  owning test helper header/source, with friend declarations in the result
  class.
- Keep handling for omitted, sentinel, or non-deterministic fields explicit in
  the generated snippet, with a short explanation when needed.

Use C++17 aggregate syntax, `repr` or existing member stream support, and no
designated initializers. Test that the generated snippet is valid and includes
the assertion and any required setup.

### Add or update focused tests

Test each comparison policy and output path that the result uses:

- Equal values pass and mismatches fail with useful diagnostics; cover each
  custom comparison policy.
- Different vector sizes fail cleanly when the result contains vectors.
- Check tolerance boundaries when tolerant comparison is supported.
- Verify that the friend `operator<<` emits the expected representation and
  friend `print_expected` emits a usable snippet.
- For an existing struct, add focused regression coverage for corrected
  omissions, field policies, or diagnostics without changing unrelated
  semantics.

Build and run the narrow affected tests using the `test-change` skill. Report
any required build configuration that prevents those tests from running.
