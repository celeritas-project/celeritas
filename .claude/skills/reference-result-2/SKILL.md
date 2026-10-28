---
name: reference-result-2
description: 'Streamlined workflow for identifying, creating, and refactoring printable Celeritas test reference-result structs for EXPECT_REF_EQ and EXPECT_REF_NEAR.'
argument-hint: 'Describe the test/file, expected fields, and comparison needs'
---

# Celeritas Reference Results

Use this workflow to replace repeated test assertions with a test-only result
struct, or to print a copy-pastable expected result. Keep the result close to
the test/helper that owns the data. Follow checked-in interfaces and preserve
existing comparison behavior.

## 1. Identify the reference result

- Start from the provided test and nearby call sites. Find the repeated values,
  their fields, and any intentional omissions, sentinels, or normalization.
- Decide whether to reuse an existing result, create a test-only result, or
  compare a library class directly. For a library class, inspect its existing
  test helper and comparison conventions before adding customization.
- Classify each field as exact or tolerant. Keep discrete fields exact; identify
  per-test tolerance overrides and any fields needing independent tolerances.
- Check `test/AssertionHelper.hh`, `test/testdetail/TestMacrosImpl.hh`, and
  `test/TestMacros.hh` for comparison dispatch and helper APIs. See
  `test/accel/HepMC3PrimaryGenerator.test.cc` for a representative result.
- Keep the implementation in the owning test/helper. For new C++ files, follow
  the `new-class` skill.

## 2. Create the reference result

### Synopsis

A reference result groups the expected values, comparison policy, and printable
output in one type. For example:

```cpp
auto result = run_test(1);
ImportXsSummary ref{{0, 0}, {0.0001, 100000000, 0.0001, 100000000}, {}};
EXPECT_REF_EQ(ref, result) << result;

// Capture updated expected values as a copy-pastable initializer and assertion.
print_expected(run_test(2));
```

### Components (declarations)

Inventory the fields first, then declare only what the result needs:

- A simple, printable result struct. Prefer POD-like fields and existing ID,
  units, and representation types. Preserve field order and public API when
  extending an existing result.
- A tolerance struct only when approximate comparisons need independently
  configurable values. Name each tolerance for its field or shared semantics.
- ADL-visible friend declarations for `operator<<`,
  `print_expected(Result const&)`, and `check_ref_eq`. Every reference-result
  class has these; `check_ref_eq` returns `void`.
- For a reusable result class, an ADL-visible `is_ref_eq` comparison policy.
  A test-local, one-off result can compare fields directly in `check_ref_eq`.

For a main-library class, place reusable comparison declarations in the
corresponding test helper. ADL may not find `celeritas::test::is_ref_eq` for a
class in the main namespace; callers may need
`using celeritas::test::is_ref_eq;`.

### Implementation (definitions)

Implement comparison once, using the existing dispatcher and field helpers:

- `EXPECT_REF_EQ` dispatches with `std::nullptr_t`; `EXPECT_REF_NEAR` dispatches
  with the supplied tolerance. Do not add result-specific `IsRefEq` overloads.
- For a reusable result, define `is_ref_eq` and have the friend `check_ref_eq`
  delegate to it. If there is no custom tolerance, provide only the
  `std::nullptr_t` policy and matching friend. If there is a tolerance, provide
  its policy and let the `std::nullptr_t` form forward with the default
  tolerance.
- For field-specific semantics, implement `check_ref_eq` field by field.
  Forward the no-tolerance overload to the default tolerance overload when
  needed, rather than duplicating comparisons. Use `is_eq` for exact scalars,
  `is_vec_eq` for exact containers, `is_vec_soft_eq` for tolerant containers,
  and `is_soft_eq` for tolerant scalars. Keep field labels and continue checking
  later fields after a mismatch. Use lower-level vector predicates only when
  wrappers cannot express the behavior, and then check sizes before traversal.
- Keep discrete fields exact. Pass each soft-compared field its matching
  tolerance member; set wider test-specific tolerances at the assertion site.
  Preserve omitted, sentinel, and normalized field behavior explicitly.
- Define small test-local results at namespace scope with inline friend
  definitions; local classes cannot define friend functions. Put substantial or
  shared definitions in the owning helper source. Include `AssertionHelper.hh`
  where a complete type is needed and `<cstddef>` for `std::nullptr_t`.

Provide printable output:

- `operator<<` emits the result representation. For field-based structs, use
  `CELER_REF_ATTR(field)` and `repr`. For aggregates, emit only a valid
  aggregate initializer in declaration order.
- `print_expected(Result const&)` uses the stream representation to print a
  copy-pastable initializer, assertion, and any required setup. Keep framing and
  `EXPECT_REF_EQ` in this function; put per-test tolerance overrides at the
  assertion call site.
- Make any omitted or non-deterministic fields explicit in the generated
  snippet, with a short explanation if needed.

After adding the result and comparison helpers, build the owning test target
before migrating assertions. Use the local configured build and the
`test-change` skill to find and run the narrow test. Fix compile or test failures
in this scaffolding before proceeding. If dependencies or configuration block
that check, report the blocker and do not continue the migration.

## 3. Refactor existing comparisons

Translate the existing test behavior into the reference result without changing
what the test means:

- Map each existing expected value to its corresponding field. Compare the
  field inventory against construction, comparator logic, and printed output;
  retain intentional omissions, sentinels, and normalization.
- Replace repeated assertions with one reference object and
  `EXPECT_REF_EQ` or `EXPECT_REF_NEAR`. Put expected values directly in that
  object with an aggregate initializer or field assignments. Avoid temporary
  or static `expected_*` arrays/vectors used only to initialize it.
- For version-dependent values, initialize common fields and assign the
  differing field directly. If only a small subset is asserted, it is
  acceptable to copy the actual result and set just those expected fields.
- Keep wider per-test tolerances at the call site and discrete values exact.
- If the goal is only to update expected values, run the test to obtain the
  actual result and call `print_expected`; review the emitted code before
  applying it. Keep the migration local to the identified test behavior.
