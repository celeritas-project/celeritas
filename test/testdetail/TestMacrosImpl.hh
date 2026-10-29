//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/TestMacrosImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>

#include "corecel/io/Repr.hh"
#include "corecel/math/SoftEqual.hh"

#include "TestMacrosTraits.hh"
#include "../AssertionHelper.hh"

namespace celeritas
{
class Constant;

namespace test
{
// Forward declare functions from bottom
template<class T>
bool is_soft_eq(
    AssertionHelper& helper, T expected, T actual, EqualOr<SoftEqual<T>> comp);

}  // namespace test

namespace testdetail
{
//---------------------------------------------------------------------------//

template<class T>
using SoftComparator_t = EqualOr<SoftEqual<T>>;

//---------------------------------------------------------------------------//
// FUNCTION DECLARATIONS
//---------------------------------------------------------------------------//

// Number of base-10 digits in an unsigned integer
int num_digits(unsigned long val);

// Return a replacement string if the given string is too long
char const* trunc_string(int digits, char const* str, char const* trunc);

// Compare two JSON objects
::testing::AssertionResult IsJsonEq(char const* expected_expr,
                                    char const* actual_expr,
                                    std::string_view expected,
                                    std::string_view actual);

extern char const expected_result_header[];
extern char const expected_result_footer[];

//---------------------------------------------------------------------------//
// CONTAINER EQUIVALENCE
//---------------------------------------------------------------------------//
//! A single index/expected/actual value
template<class T1, class T2>
struct FailedValue
{
    using size_type = std::size_t;
    using first_type = T1;
    using second_type = T2;

    size_type index;
    first_type expected;
    second_type actual;
};

// Two Container Traits
template<class C1, class C2>
struct TCT
{
    template<class C>
    using value_type_ = typename ContTraits<C>::value_type;
    template<class C>
    using nc_value_type_ = std::remove_const_t<value_type_<C>>;

    using first_type = nc_value_type_<C1>;
    using second_type = nc_value_type_<C2>;

    using common_type = std::common_type_t<first_type, second_type>;

    using VecFailedValue = std::vector<FailedValue<first_type, second_type>>;
};

// Failed value iterator traits
template<class Iter1, class Iter2>
struct FVIT
{
    template<class I>
    using value_type_ = typename std::iterator_traits<I>::value_type;
    template<class I>
    using nc_value_type_ = std::remove_const_t<value_type_<I>>;

    using first_type = nc_value_type_<Iter1>;
    using second_type = nc_value_type_<Iter2>;

    using type = FailedValue<first_type, second_type>;
    using Vec_t = std::vector<type>;
};

//---------------------------------------------------------------------------//
// CELER_ATTR_REPR IMPLEMENTATION
//---------------------------------------------------------------------------//

//! Print an attribute with CELER_ATTR_REPR
template<class T>
struct StreamableAttr
{
    char const* name;
    T const& value;

    friend std::ostream& operator<<(std::ostream& os,
                                    StreamableAttr const& attr)
    {
        return os << "/* " << attr.name << " = */ " << repr(attr.value);
    }
};

// CTAD for C++17
template<class T>
StreamableAttr(char const*, T const&) -> StreamableAttr<T>;

//---------------------------------------------------------------------------//
// EXPECTED RESULT PRINTING
//---------------------------------------------------------------------------//

/*!
 * Print expected values of a container
 *
 * \todo deprecate: use repr
 */
template<class Container>
std::enable_if_t<IsContainer_v<Container>, void> print_expected(
    Container const& data)
{
    using RT = ReprTraits<Container>;
    using VRT = ReprTraits<typename RT::value_type>;

    std::cout << "static ";
    VRT::print_type(std::cout, "const ref");
    std::cout << "[] = ";

    std::ios orig_state(nullptr);
    orig_state.copyfmt(std::cout);
    VRT::init(std::cout);
    RT::print_value(std::cout, data);
    std::cout.copyfmt(orig_state);

    std::cout << ";\n";
}

//! Print expected values for a reference result with header/footer
template<class Result>
std::enable_if_t<IsRefResult_v<Result>, void> print_expected(
    Result const& result)
{
    std::cout << expected_result_header << Result::class_str
              << " ref = " << result << ";\n"
              << expected_result_footer;
}

//---------------------------------------------------------------------------//
// CHECK_REF_EQ
//---------------------------------------------------------------------------//

/*!
 * Compare containers of reference values using a tolerance.
 */
template<class ContainerE, class ContainerA, class Tol>
std::enable_if_t<IsContainer_v<ContainerE> && IsContainer_v<ContainerA>, bool>
check_ref_eq(::celeritas::test::AssertionHelper& helper,
             ContainerE const& val1,
             ContainerA const& val2,
             Tol&& tol)
{
    using std::begin;
    using std::end;

    if (!helper.equal_size(std::distance(begin(val1), end(val1)),
                           std::distance(begin(val2), end(val2))))
    {
        return false;
    }

    // Check each element
    auto iter1 = begin(val1);
    auto iter2 = begin(val2);
    size_type i = 0;
    size_type failures = 0;
    constexpr size_type max_printable_failures = 10;
    for (auto end1 = end(val1); iter1 != end1; ++i, ++iter1, ++iter2)
    {
        if (!check_ref_eq(helper, *iter1, *iter2, tol))
        {
            if (failures++ < max_printable_failures)
            {
                helper.fail() << "\n(Failed in element " << i << " of "
                              << helper.actual_expr() << ")";
            }
        }
    }
    if (failures > max_printable_failures)
    {
        helper.fail() << "(Suppressed " << failures - max_printable_failures
                      << " additional failures)";
    }
    return failures == 0;
}

template<class ContainerE, class ContainerA>
std::enable_if_t<IsContainer_v<ContainerE> && IsContainer_v<ContainerA>, bool>
check_ref_eq(::celeritas::test::AssertionHelper& helper,
             ContainerE const& val1,
             ContainerA const& val2)
{
    using std::begin;
    using std::end;

    if (!helper.equal_size(std::distance(begin(val1), end(val1)),
                           std::distance(begin(val2), end(val2))))
    {
        return false;
    }

    auto iter1 = begin(val1);
    auto iter2 = begin(val2);
    size_type i = 0;
    size_type failures = 0;
    constexpr size_type max_printable_failures = 10;
    for (auto end1 = end(val1); iter1 != end1; ++i, ++iter1, ++iter2)
    {
        if (!check_ref_eq(helper, *iter1, *iter2))
        {
            if (failures++ < max_printable_failures)
            {
                helper.fail() << "\n(Failed in element " << i << " of "
                              << helper.actual_expr() << ")";
            }
        }
    }
    if (failures > max_printable_failures)
    {
        helper.fail() << "(Suppressed " << failures - max_printable_failures
                      << " additional failures)";
    }
    return failures == 0;
}

//---------------------------------------------------------------------------//
// VECTOR FAILURE IMPLEMENTATION
//---------------------------------------------------------------------------//

/*!
 * Print failure results.
 */
template<class T1, class T2>
std::string failure_msg(char const* expected_expr,
                        char const* actual_expr,
                        std::vector<FailedValue<T1, T2>> const& failures)
{
    using RT1 = ReprTraits<T1>;
    using RT2 = ReprTraits<T2>;
    using std::setw;

    // Calculate how many digits we need to space out
    int idig = num_digits(static_cast<unsigned long>(failures.back().index));
    int vdig = 16;

    // Construct our own stringstream because google test ignores setw
    std::ostringstream os;
    RT2::init(os);
    RT1::init(os);

    // Print column headers (unless expected/actual is too long)
    os << setw(idig) << 'i' << ' ' << setw(vdig)
       << trunc_string(vdig, expected_expr, "EXPECTED") << ' ' << setw(vdig)
       << trunc_string(vdig, actual_expr, "ACTUAL") << '\n';

    // Loop through failed indices and print values
    for (auto const& f : failures)
    {
        os << setw(idig) << f.index << ' ' << setw(vdig);
        RT1::print_value(os, f.expected);
        os << ' ' << setw(vdig);
        RT2::print_value(os, f.actual);
        os << '\n';
    }
    return os.str();
}

/*!
 * Print failure results for floating point comparisons.
 */
template<class T1, class T2>
std::string float_failure_msg(char const* expected_expr,
                              char const* actual_expr,
                              std::vector<FailedValue<T1, T2>> const& failures,
                              double abs_thresh)
{
    using std::setprecision;
    using std::setw;

    // Calculate how many digits we need to space out the index
    int idig = num_digits(static_cast<unsigned long>(failures.back().index));
    int vdig = std::max(std::numeric_limits<T1>::digits10,
                        std::numeric_limits<T2>::digits10);

    // Construct our own stringstream because google test ignores setw
    std::ostringstream os;
    os << setprecision(vdig);
    vdig += 4;

    // Try to use user-given expressions for headers, but fall back if the
    // column length is exceeded
    std::string e_expr(expected_expr);
    std::string a_expr(actual_expr);

    os << setw(idig) << 'i' << ' ' << setw(vdig)
       << trunc_string(vdig, expected_expr, "EXPECTED") << setw(vdig)
       << trunc_string(vdig, actual_expr, "ACTUAL") << setw(vdig)
       << "Difference" << '\n';

    // Loop through failed indices and print values
    for (auto const& f : failures)
    {
        os << setw(idig) << f.index << ' ' << setw(vdig) << f.expected << ' '
           << setw(vdig) << f.actual << ' ' << setw(vdig);

        if (std::isinf(f.expected))
        {
            os << "---";
        }
        else if (std::fabs(f.expected) > abs_thresh)
        {
            os << (f.actual - f.expected) / f.expected;
        }
        else
        {
            os << f.actual - f.expected;
        }
        os << '\n';
    }
    return os.str();
}

/*!
 * Compare a range of values.
 */
template<class Iter1, class Iter2, class BinaryOp>
::testing::AssertionResult is_range_eq(
    Iter1 e_iter,
    Iter1 e_end,
    char const* expected_expr,
    Iter2 a_iter,
    Iter2 a_end,
    char const* actual_expr,
    typename FVIT<Iter1, Iter2>::Vec_t& failures,
    BinaryOp comp)
{
    using size_type = std::size_t;
    auto expected_size = std::distance(e_iter, e_end);
    auto actual_size = std::distance(a_iter, a_end);

    // First, check that the sizes are equal
    if (expected_size != actual_size)
    {
        ::testing::AssertionResult failure = ::testing::AssertionFailure();

        failure << " Size of: " << actual_expr << "\n  Actual: " << actual_size
                << "\nExpected: " << expected_expr
                << ".size()\nWhich is: " << expected_size << '\n';
        return failure;
    }

    // Save start iterator in order to save index
    Iter1 const e_begin = e_iter;

    for (; e_iter != e_end; ++e_iter, ++a_iter)
    {
        if (!comp(*e_iter, *a_iter))
        {
            size_type i = e_iter - e_begin;
            failures.push_back({i, *e_iter, *a_iter});
        }
    }

    if (failures.empty())
    {
        return ::testing::AssertionSuccess();
    }

    ::testing::AssertionResult result = ::testing::AssertionFailure();
    result << "Values in: " << actual_expr << "\n Expected: " << expected_expr
           << '\n'
           << failures.size() << " of " << expected_size << " elements differ";
    // Chop out the middle failures so we can see the beginning/end but not be
    // overwhelmed for large outputs. When truncation happens, trunc_size is
    // always 20, but it's written with `min` to avoid a dozen different false
    // positives from GCC range checking when the expression is consteval.
    auto const trunc_size = std::min<std::size_t>(failures.size(), 20);
    if (failures.size() > 2 * trunc_size)
    {
        failures.erase(failures.begin() + trunc_size,
                       failures.end() - trunc_size);
        result << " (truncating by removing all but the first and last "
               << trunc_size << ")";
    }
    result << '\n';
    return result;
}

//---------------------------------------------------------------------------//
// GTEST MACRO IMPLEMENTATION
//---------------------------------------------------------------------------//

/*!
 * Predicate for relative error soft equivalence.
 */
template<class VE, class VA>
::testing::AssertionResult IsSoftEquiv(char const* expected_expr,
                                       char const* actual_expr,
                                       const VE& expected,
                                       const VA& actual)
{
    static_assert(can_soft_equiv<VE, VA>, "Invalid types for soft equivalence");

    using common_type = CommonPrecision_t<VE, VA>;
    test::AssertionHelper result{expected_expr, actual_expr};
    is_soft_eq(result,
               static_cast<common_type>(expected),
               static_cast<common_type>(actual),
               SoftComparator_t<common_type>());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Predicate for relative error soft equivalence.
 */
template<class VE, class VA, class T>
::testing::AssertionResult IsSoftEquiv(char const* expected_expr,
                                       char const* actual_expr,
                                       char const*,
                                       VE const& expected,
                                       VA const& actual,
                                       T&& cmp_or_tol)
{
    static_assert(can_soft_equiv<VE, VA>, "Invalid types for soft equivalence");

    using common_type = CommonPrecision_t<VE, VA>;
    test::AssertionHelper result{expected_expr, actual_expr};
    is_soft_eq(result,
               static_cast<common_type>(expected),
               static_cast<common_type>(actual),
               SoftComparator_t<common_type>(std::forward<T>(cmp_or_tol)));
    return result;
}

/*!
 * Compare vectors with soft equivalence.
 *
 * This signature uses the default tolerance for the appropriate floating point
 * operations.
 */
template<class ContainerE, class ContainerA, class BinaryOp>
::testing::AssertionResult IsVecSoftEquivImpl(ContainerE const& expected,
                                              char const* expected_expr,
                                              ContainerA const& actual,
                                              char const* actual_expr,
                                              BinaryOp comp)
{
    if constexpr (IsNestedContainer_v<ContainerE>
                  && IsNestedContainer_v<ContainerA>)
    {
        // Handle nested containers recursively
        auto exp_size = std::distance(std::begin(expected), std::end(expected));
        auto act_size = std::distance(std::begin(actual), std::end(actual));

        if (exp_size != act_size)
        {
            ::testing::AssertionResult failure = ::testing::AssertionFailure();

            failure << " Size of: " << actual_expr
                    << "\n  Actual: " << act_size
                    << "\nExpected: " << expected_expr
                    << ".size()\nWhich is: " << exp_size << '\n';
            return failure;
        }

        for (int i = 0; i < exp_size; ++i)
        {
            auto result = IsVecSoftEquivImpl(
                expected[i], expected_expr, actual[i], actual_expr, comp);
            if (!result)
            {
                return result;
            }
        }
        return ::testing::AssertionSuccess();
    }
    else
    {
        using Traits_t = TCT<ContainerE, ContainerA>;
        using Failed_t = FailedValue<typename Traits_t::first_type,
                                     typename Traits_t::second_type>;
        std::vector<Failed_t> failures;

        ::testing::AssertionResult result = is_range_eq(std::begin(expected),
                                                        std::end(expected),
                                                        expected_expr,
                                                        std::begin(actual),
                                                        std::end(actual),
                                                        actual_expr,
                                                        failures,
                                                        comp);

        if (!result)
        {
            if (failures.empty())
            {
                // Size was different; print the actual vector
                result << "Actual values: " << repr(actual) << ";\n";
            }
            else
            {
                // Inform user of failing tolerance
                result << "by " << comp.rel() << " relative error or "
                       << comp.abs() << " absolute error\n";
                // Print indices that were different
                result << float_failure_msg(
                    expected_expr, actual_expr, failures, comp.abs());
            }
        }
        return result;
    }
}

/*!
 * Compare two containers.
 */
template<class ContainerE, class ContainerA>
::testing::AssertionResult IsVecEq(char const* expected_expr,
                                   char const* actual_expr,
                                   ContainerE const& expected,
                                   ContainerA const& actual)
{
    if constexpr (IsNestedContainer_v<ContainerE>
                  && IsNestedContainer_v<ContainerA>)
    {
        // Handle nested containers recursively
        auto exp_size = std::distance(std::begin(expected), std::end(expected));
        auto act_size = std::distance(std::begin(actual), std::end(actual));

        if (exp_size != act_size)
        {
            ::testing::AssertionResult failure = ::testing::AssertionFailure();

            failure << " Size of: " << actual_expr
                    << "\n  Actual: " << act_size
                    << "\nExpected: " << expected_expr
                    << ".size()\nWhich is: " << exp_size << '\n';
            return failure;
        }

        for (int i = 0; i < exp_size; ++i)
        {
            auto result
                = IsVecEq(expected_expr, actual_expr, expected[i], actual[i]);
            if (!result)
            {
                return result;
            }
        }
        return ::testing::AssertionSuccess();
    }
    else
    {
        using Traits_t = TCT<ContainerE, ContainerA>;

        typename Traits_t::VecFailedValue failures;

        ::testing::AssertionResult result
            = is_range_eq(std::begin(expected),
                          std::end(expected),
                          expected_expr,
                          std::begin(actual),
                          std::end(actual),
                          actual_expr,
                          failures,
                          std::equal_to<typename Traits_t::common_type>());

        if (!result)
        {
            if (failures.empty())
            {
                // Size was different; print the actual vector
                result << "Actual values: " << repr(actual) << ";\n";
            }
            else
            {
                // Print indices that were different
                result << failure_msg(expected_expr, actual_expr, failures);
            }
        }

        return result;
    }
}

/*!
 * Compare two containers using soft equivalence.
 */
template<class ContainerE, class ContainerA>
::testing::AssertionResult IsVecSoftEquiv(char const* expected_expr,
                                          char const* actual_expr,
                                          ContainerE const& expected,
                                          ContainerA const& actual)
{
    using VE = ScalarValueType_t<ContainerE>;
    using VA = ScalarValueType_t<ContainerA>;

    static_assert(can_soft_equiv<VE, VA>, "Invalid types for soft equivalence");
    using common_type = CommonPrecision_t<VE, VA>;

    // Construct with automatic or specified tolerances
    return IsVecSoftEquivImpl(expected,
                              expected_expr,
                              actual,
                              actual_expr,
                              SoftComparator_t<common_type>());
}

/*!
 * Compare two containers using soft equivalence.
 *
 * This signature uses the default tolerance for the appropriate floating point
 * operations.
 */
template<class ContainerE, class ContainerA, class T>
::testing::AssertionResult IsVecSoftEquiv(char const* expected_expr,
                                          char const* actual_expr,
                                          char const*,
                                          ContainerE const& expected,
                                          ContainerA const& actual,
                                          T&& cmp_or_tol)
{
    using VE = ScalarValueType_t<ContainerE>;
    using VA = ScalarValueType_t<ContainerA>;

    static_assert(can_soft_equiv<VE, VA>, "Invalid types for soft equivalence");
    using common_type = CommonPrecision_t<VE, VA>;

    // Construct with given tolerance
    return IsVecSoftEquivImpl(
        expected,
        expected_expr,
        actual,
        actual_expr,
        SoftComparator_t<common_type>(std::forward<T>(cmp_or_tol)));
}

/*!
 * Dispatch reference comparison without a special tolerance.
 */
template<class E, class A>
::testing::AssertionResult IsRefEq(
    char const* expr1, char const* expr2, E const& expected, A const& actual)
{
    ::celeritas::test::AssertionHelper helper{expr1, expr2};
    check_ref_eq(helper, expected, actual);
    return helper;
}

/*!
 * Dispatch reference comparison using a specified tolerance.
 */
template<class E, class A, class T>
::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   char const* tol_expr,
                                   E const& expected,
                                   A const& actual,
                                   T const& tol)
{
    ::celeritas::test::AssertionHelper helper{expr1, expr2, tol_expr};
    check_ref_eq(helper, expected, actual, tol);
    return helper;
}

//---------------------------------------------------------------------------//
}  // namespace testdetail

namespace test
{
//---------------------------------------------------------------------------//
// Note these are in the same namespace as assertion helper. This should be
// refactored to clean up recursive testing.
//---------------------------------------------------------------------------//

// Expose soft comparator type alias (to be refactored)
using testdetail::SoftComparator_t;

/*!
 * Perform a soft equality comparison, appending error details on failure.
 */
template<class T>
bool is_eq(AssertionHelper& helper, char const* attrname, T expected, T actual)
{
    if (expected == actual)
    {
        return true;
    }

    helper.fail() << "Expected " << attrname << ": " << repr(expected)
                  << " but got " << repr(actual);
    return false;
}

/*!
 * Perform a soft equality comparison, appending error details on failure.
 */
template<class T>
bool is_soft_eq(
    AssertionHelper& helper, T expected, T actual, SoftComparator_t<T> comp)
{
    if (comp(expected, actual))
    {
        return true;
    }

    auto&& msg = helper.fail();
    msg << "Value of: " << helper.actual_expr()
        << "\n  Actual: " << repr(actual)
        << "\nExpected: " << helper.expected_expr()
        << "\nWhich is: " << repr(expected);

    if (std::fabs(actual - expected) > comp.abs())
    {
        msg << '\n'
            << "- absolute error " << repr(actual - expected)
            << " exceeds tolerance " << repr(comp.abs());
    }
    if (expected != 0
        && std::fabs(actual - expected) > std::fabs(expected) * comp.rel())

    {
        msg << "\n- relative error " << repr(actual / expected - 1)
            << " exceeds tolerance " << repr(comp.rel());
    }
    return false;
}

/*!
 * Perform vector equality comparison, appending error details on failure.
 */
template<class ContainerE, class ContainerA>
bool is_vec_eq(AssertionHelper& helper,
               char const* field,
               ContainerE const& expected,
               ContainerA const& actual)
{
    if (!helper.equal_size(expected.size(), actual.size()))
    {
        helper.fail() << "  " << field << ": " << helper.actual_expr() << " = "
                      << repr(actual);
        return false;
    }

    if (auto r = ::celeritas::testdetail::IsVecEq(
            helper.expected_expr(), helper.actual_expr(), expected, actual);
        !r)
    {
        helper.fail() << "  " << field << ": " << r.message();
        return false;
    }

    return true;
}

/*!
 * Perform vector soft equality comparison with a tolerance, appending error
 * details on failure.
 */
template<class ContainerE, class ContainerA, class Tol>
bool is_vec_soft_eq(AssertionHelper& helper,
                    char const* field,
                    ContainerE const& expected,
                    ContainerA const& actual,
                    Tol const& tol)
{
    using namespace ::celeritas::testdetail;
    if (!helper.equal_size(expected.size(), actual.size()))
    {
        helper.fail() << "  " << field << ": " << helper.actual_expr() << " = "
                      << repr(actual);
        return false;
    }
    if (auto r = IsVecSoftEquiv(helper.expected_expr(),
                                helper.actual_expr(),
                                "",
                                expected,
                                actual,
                                tol);
        !r)
    {
        helper.fail() << "  " << field << ": " << r.message();
        return false;
    }

    return true;
}

/*!
 * Perform vector soft equality comparison, appending error details on failure.
 */
template<class ContainerE, class ContainerA>
inline bool is_vec_soft_eq(AssertionHelper& helper,
                           char const* field,
                           ContainerE const& expected,
                           ContainerA const& actual)
{
    using namespace ::celeritas::testdetail;
    // Use default tolerance
    using VE = ScalarValueType_t<ContainerE>;
    using VA = ScalarValueType_t<ContainerA>;
    static_assert(can_soft_equiv<VE, VA>, "Invalid types for soft equivalence");
    using common_type = CommonPrecision_t<VE, VA>;
    return is_vec_soft_eq(
        helper, field, expected, actual, SoftComparator_t<common_type>());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
