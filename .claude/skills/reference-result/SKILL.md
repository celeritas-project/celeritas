---
name: reference-result
description: 'Identify, create, and refactor printable Celeritas test reference-result structs for EXPECT_REF_EQ and EXPECT_REF_NEAR.'
argument-hint: 'Describe the test/file, expected fields, and comparison needs'
---

# Celeritas Reference Results

Use this workflow to replace repeated test assertions with a test-only result
struct, or to print a copy-pastable expected result. Keep the result close to
the test/helper that owns the data. Follow checked-in interfaces and preserve
existing comparison behavior.

A reference result groups the expected values, comparison policy, and printable
output in one type. Use the test-local `EventResult` in
`test/accel/HepMC3PrimaryGenerator.test.cc` as the canonical example: it keeps
the result fields together, uses `Array<double, 3>` for positions and
directions, defines its tolerance policy alongside the data, and provides
printable output and comparison friends after the fields.

The result type is summarized here:
```cpp
struct EventResult
{
  std::vector<Real3> pos;
  std::vector<Real3> dir;
  std::vector<double> energy;
  std::vector<int> vtx;

  struct Tol
  {
    double dir{Test::coarse_eps};
    double energy{Test::fine_eps};
  };

  static constexpr char class_str[] = "EventResult";

  friend std::ostream& operator<<(std::ostream& os, EventResult const& ref)
  {
    return os << '{' << CELER_ATTR_REPR(ref, pos) << ", "
          << CELER_ATTR_REPR(ref, dir) << ", "
          << CELER_ATTR_REPR(ref, energy) << ", "
          << CELER_ATTR_REPR(ref, vtx) << '}';
  }

  friend bool is_ref_eq(AssertionHelper& helper,
               EventResult const& expected,
               EventResult const& actual,
               Tol const& tol)
  {
    RefComparer<EventResult> compare{helper, expected, actual};
    CELER_INVOKE_ATTR(compare.is_soft_eq, EventResult, pos);
    CELER_INVOKE_ATTR(compare.is_soft_eq, EventResult, dir, tol.dir);
    CELER_INVOKE_ATTR(compare.is_soft_eq, EventResult, energy, tol.energy);
    CELER_INVOKE_ATTR(compare.is_eq, EventResult, vtx);
    return static_cast<bool>(compare);
  }

  friend bool is_ref_eq(AssertionHelper& helper,
               EventResult const& expected,
               EventResult const& actual)
  {
    return is_ref_eq(helper, expected, actual, {});
  }
};

auto result = run_test();

// Capture updated expected values as a copy-pastable initializer and assertion.
print_expected(result);

EventResult ref{};
EXPECT_REF_EQ(ref, result);
// OR
EventResult::Tol tol{};
EXPECT_REF_NEAR(ref, result, tol);
```

## 1. Identify the reference result

- Start from the provided test and nearby call sites. Find the repeated values,
  their fields, and any intentional omissions, sentinels, or normalization.
- Decide whether to reuse an existing result, create a test-only result, or
  compare a library class directly. For a library class, inspect its existing
  test helper and comparison conventions before adding customization.
- Classify each field as exact or tolerant. Keep discrete fields exact; identify
  per-test tolerance overrides and any fields needing independent tolerances.
- See `test/accel/HepMC3PrimaryGenerator.test.cc` for a representative
  test-local result or `test/geocel/RefUtils.hh` for a library-based result.
  Review `test/AssertionHelper.hh`, `test/testdetail/TestMacrosImpl.hh`, and
  `test/TestMacros.hh` for comparison dispatch and helper APIs if further details are needed.
- Keep the implementation in the owning test/helper for local tests or in a new file
  if reusing across multiple tests.

## 2. Declare the reference result

Inventory the fields first, then declare only what the result needs:

- A printable result struct with the data fields first and reference-result
  functions after them. Non-POD and custom domain fields are allowed; use
  useful types such as `Array<double, 3>` (`Real3`) for positions and directions
  rather than flattening them solely to make the result POD-like. Preserve
  field order and public API when extending an existing result.
- A tolerance struct only when approximate comparisons need independently
  configurable values. Use plain floating-point members initialized to the
  normal default tolerance (for example, `Test::fine_eps`); do not use optional
  members to select between default and custom comparison behavior. Nest the
  tolerance struct in the result when it is specific to that result. Name each
  tolerance for its field or shared semantics.
- ADL-visible friends for `operator<<` and `is_ref_eq`. Return `bool` from
  `is_ref_eq`, especially when the result is reusable or can be an element
  of a container. Keep small test-local implementations inline in the test
  header/source; for a reusable test library, declare the friend in the result
  type and define it in the corresponding `.cc` file.
- The generic `print_expected` overload is imported into the global namespace
  by `test/TestMacros.hh`; do not add a result-specific member, friend, or
  free-function forwarding wrapper.

For a main-library class, place reusable `is_ref_eq` declarations in the
corresponding test helper. Ensure the comparison friend is discoverable by ADL
from the result type; callers may need a using-declaration for test helper
functions when the type is in the main namespace.

## 3. Implement the reference result.

Implement comparison once, using the existing dispatcher and field helpers:

- `EXPECT_REF_EQ` dispatches to `is_ref_eq` without a tolerance;
  `EXPECT_REF_NEAR` dispatches with the supplied tolerance. Do not add
  result-specific `IsRefEq` overloads.
- For a result with a nested tolerance type, define a four-argument friend that
  accepts `Tol const&`, then define the three-argument friend to create and
  forward with the default tolerance (`return is_ref_eq(..., {});`). Do not
  use a default argument of the nested tolerance type on the friend; that
  default argument is prohibited in this context.
- For a result with no tolerance policy, a four-argument friend may use
  `std::nullptr_t = {}` as its final parameter. This supports ordinary
  three-argument comparison while also allowing container comparison dispatch
  to pass an explicit no-tolerance argument; include `<cstddef>` for
  `std::nullptr_t`.
- For field-specific semantics, implement `is_ref_eq` field by field.
  Pass tolerance members directly to soft-comparison helpers, including the
  defaulted values. Use `is_eq` for exact scalars, `is_vec_eq` for exact
  containers, `is_vec_soft_eq` for tolerant containers, and `is_soft_eq` for
  tolerant scalars. Keep field labels and continue checking later fields after
  a mismatch. For a vector of reference-result objects, call the
  container-overloaded `is_ref_eq(helper, expected.items, actual.items)` (or
  pass its tolerance when applicable). It recursively dispatches to each
  element's ADL-visible friend; do not write a second element-comparison loop.
  Use lower-level vector predicates only when wrappers cannot express the
  behavior, and then check sizes before traversal.
- Keep discrete fields exact. Pass each soft-compared field its matching
  tolerance member; set wider test-specific tolerances at the assertion site.
  Preserve omitted, sentinel, and normalized field behavior explicitly.
- Define small test-local results at namespace scope; local classes cannot
  define friend functions. Put substantial or shared comparison definitions in
  the owning reusable test library's `.cc` file. Include `AssertionHelper.hh`
  where a complete type is needed.

Provide printable output:

- Give the result type a `class_str` and an ADL-visible `operator<<`. For a
  field-based aggregate, emit a braced initializer in declaration order using
  `CELER_ATTR_REPR(ref, field)` for each field, as in
  `test/accel/HepMC3PrimaryGenerator.test.cc`'s `EventResult`. This prints
  each field's name as a comment alongside its `repr` value while keeping the
  output valid as an aggregate initializer. Do not use `CELER_REF_ATTR` here:
  it prints assignment statements, not an initializer.
- Call the imported `print_expected(result)` overload from test code. It uses
  the stream representation to print a copy-pastable `Type ref = {...};`
  initializer and assertion. Keep per-test tolerance overrides at the
  assertion call site.
- Use a separate printer only when the output needs genuinely custom setup or
  domain-specific assertions; do not add a pass-through wrapper around the
  generic overload.
- Make any omitted or non-deterministic fields explicit in the generated
  snippet, with a short explanation if needed.

After adding the result and comparison helpers, build the owning test target
before migrating assertions. Use the local configured build and the
`test-change` skill to find and run the narrow test. Fix compile or test failures
in this scaffolding before proceeding. If dependencies or configuration block
that check, report the blocker and do not continue the migration.

## 4. Refactor existing comparisons

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
