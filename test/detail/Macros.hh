//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file detail/Macros.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include <vector>
#include <gtest/gtest.h>

#include "celeritas_config.h"
#include "corecel/Macros.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/SoftEqual.hh"

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//

//! Container equality macro
#define EXPECT_VEC_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(::celeritas_test::detail::IsVecEq, expected, actual)

//! Soft equivalence macro
#define EXPECT_SOFT_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(                 \
        ::celeritas_test::detail::IsSoftEquiv, expected, actual)

//! Soft equivalence macro with relative error
#define EXPECT_SOFT_NEAR(expected, actual, rel_error) \
    EXPECT_PRED_FORMAT3(                              \
        ::celeritas_test::detail::IsSoftEquiv, expected, actual, rel_error)

//! Container soft equivalence macro
#define EXPECT_VEC_SOFT_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(                     \
        ::celeritas_test::detail::IsVecSoftEquiv, expected, actual)

//! Container soft equivalence macro with relative error
#define EXPECT_VEC_NEAR(expected, actual, rel_error) \
    EXPECT_PRED_FORMAT3(                             \
        ::celeritas_test::detail::IsVecSoftEquiv, expected, actual, rel_error)

//! Container soft equivalence macro with relative and absolute error
#define EXPECT_VEC_CLOSE(expected, actual, rel_error, abs_thresh) \
    EXPECT_PRED_FORMAT4(::celeritas_test::detail::IsVecSoftEquiv, \
                        expected,                                 \
                        actual,                                   \
                        rel_error,                                \
                        abs_thresh)

//! Print the given container as an array for regression testing
#define PRINT_EXPECTED(data) \
    ::celeritas_test::detail::print_expected(data, #data)

//! Construct a test name that is disabled when assertions are enabled
#if CELERITAS_DEBUG
#    define TEST_IF_CELERITAS_DEBUG(name) name
#else
#    define TEST_IF_CELERITAS_DEBUG(name) DISABLED_##name
#endif

//! Construct a test name that is disabled when CUDA/HIP are disabled
#if CELER_USE_DEVICE
#    define TEST_IF_CELER_DEVICE(name) name
#else
#    define TEST_IF_CELER_DEVICE(name) DISABLED_##name
#endif

//! Construct a test name that is disabled when JSON is disabled
#if CELERITAS_USE_GEANT4
#    define TEST_IF_CELERITAS_GEANT(name) name
#else
#    define TEST_IF_CELERITAS_GEANT(name) DISABLED_##name
#endif

//! Construct a test name that is disabled when JSON is disabled
#if CELERITAS_USE_JSON
#    define TEST_IF_CELERITAS_JSON(name) name
#else
#    define TEST_IF_CELERITAS_JSON(name) DISABLED_##name
#endif

//! Construct a test name that is disabled when ROOT is disabled
#if CELERITAS_USE_ROOT
#    define TEST_IF_CELERITAS_USE_ROOT(name) name
#else
#    define TEST_IF_CELERITAS_USE_ROOT(name) DISABLED_##name
#endif

namespace celeritas_test
{
namespace detail
{
//---------------------------------------------------------------------------//
// FUNCTION DECLARATIONS
//---------------------------------------------------------------------------//

// Number of base-10 digits in an unsigned integer
int num_digits(unsigned long val);

// Return a replacement string if the given string is too long
const char*
trunc_string(unsigned int digits, const char* str, const char* trunc);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
//! Whether soft equivalence can be performed on the given types.
template<class T1, class T2>
constexpr bool can_soft_equiv()
{
    return (std::is_floating_point<T1>::value
            || std::is_floating_point<T2>::value)
           && std::is_convertible<T1, T2>::value;
}

//---------------------------------------------------------------------------//
/*!
 * Get a "least common denominator" for soft comparisons.
 */
template<class T1, class T2>
struct SoftPrecisionType
{
    using type = std::common_type_t<T1, T2>;
};

// When comparing doubles to floats, use the floating point epsilon for
// comparison
template<>
struct SoftPrecisionType<double, float>
{
    using type = float;
};

template<>
struct SoftPrecisionType<float, double>
{
    using type = float;
};

//---------------------------------------------------------------------------//
//! Compare a range of values.
template<class BinaryOp>
::testing::AssertionResult
IsSoftEquivImpl(typename BinaryOp::value_type expected,
                const char*                   expected_expr,
                typename BinaryOp::value_type actual,
                const char*                   actual_expr,
                BinaryOp                      comp)
{
    using value_type = typename BinaryOp::value_type;

    if (comp(expected, actual))
    {
        return ::testing::AssertionSuccess();
    }

    // Failed: print nice error message
    ::testing::AssertionResult result = ::testing::AssertionFailure();

    result << "Value of: " << actual_expr << "\n  Actual: " << actual
           << "\nExpected: " << expected_expr << "\nWhich is: " << expected
           << '\n';

    celeritas::SoftZero<value_type> is_soft_zero(comp.abs());
    if (is_soft_zero(expected))
    {
        // Avoid divide by zero errors
        result << "(Absolute error " << actual - expected
               << " exceeds tolerance " << comp.abs() << ")";
    }
    else
    {
        result << "(Relative error " << (actual - expected) / expected
               << " exceeds tolerance " << comp.rel() << ")";
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Predicate for relative error soft equivalence.
 */
template<class Value_E, class Value_A>
::testing::AssertionResult IsSoftEquiv(const char* expected_expr,
                                       const char* actual_expr,
                                       Value_E     expected,
                                       Value_A     actual)
{
    static_assert(can_soft_equiv<Value_E, Value_A>(),
                  "Invalid types for soft equivalence");

    // Construct with automatic or specified tolerances
    using ValueT   = typename SoftPrecisionType<Value_E, Value_A>::type;
    using BinaryOp = celeritas::SoftEqual<ValueT>;

    return IsSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp());
}

//---------------------------------------------------------------------------//
/*!
 * Predicate for relative error soft equivalence.
 */
template<class Value_E, class Value_A>
::testing::AssertionResult IsSoftEquiv(const char* expected_expr,
                                       const char* actual_expr,
                                       const char*,
                                       Value_E expected,
                                       Value_A actual,
                                       double  rel)
{
    static_assert(can_soft_equiv<Value_E, Value_A>(),
                  "Invalid types for soft equivalence");

    // Construct with automatic or specified tolerances
    using ValueT   = typename SoftPrecisionType<Value_E, Value_A>::type;
    using BinaryOp = celeritas::SoftEqual<ValueT>;

    return IsSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp(rel));
}

//---------------------------------------------------------------------------//
// CONTAINER EQUIVALENCE
//---------------------------------------------------------------------------//
//! A single index/expected/actual value
template<class T1, class T2>
struct FailedValue
{
    using size_type   = std::size_t;
    using first_type  = T1;
    using second_type = T2;

    size_type   index;
    first_type  expected;
    second_type actual;
};

// Two Container Traits
template<class C1, class C2>
struct TCT
{
    template<class C>
    using value_type_ = typename celeritas::ContTraits<C>::value_type;
    template<class C>
    using nc_value_type_ = typename std::remove_const<value_type_<C>>::type;

    using first_type  = nc_value_type_<C1>;
    using second_type = nc_value_type_<C2>;

    using common_type =
        typename std::common_type<first_type, second_type>::type;

    using Failed_Value_t = FailedValue<first_type, second_type>;
    using Failed_Vec_t   = std::vector<Failed_Value_t>;
};

// Failed value iterator traits
template<class Iter1, class Iter2>
struct FVIT
{
    template<class I>
    using value_type_ = typename std::iterator_traits<I>::value_type;
    template<class I>
    using nc_value_type_ = typename std::remove_const<value_type_<I>>::type;

    using first_type  = nc_value_type_<Iter1>;
    using second_type = nc_value_type_<Iter2>;

    using type  = FailedValue<first_type, second_type>;
    using Vec_t = std::vector<type>;
};

//---------------------------------------------------------------------------//
/*!
 * Compare a range of values.
 */
template<class Iter1, class Iter2, class BinaryOp>
::testing::AssertionResult
IsRangeEqImpl(Iter1                               e_iter,
              Iter1                               e_end,
              const char*                         expected_expr,
              Iter2                               a_iter,
              Iter2                               a_end,
              const char*                         actual_expr,
              typename FVIT<Iter1, Iter2>::Vec_t& failures,
              BinaryOp                            comp)
{
    using size_type         = std::size_t;
    size_type expected_size = std::distance(e_iter, e_end);
    size_type actual_size   = std::distance(a_iter, a_end);

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
    const Iter1 e_begin = e_iter;

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
    if (failures.size() > 40)
    {
        result << " (truncating by removing all but the first and last 20)";
        failures.erase(failures.begin() + 20, failures.end() - 20);
    }
    result << '\n';
    return result;
}

//-------------------------------------------------------------------------//
/*!
 * Compare vectors with soft equivalence.
 *
 * This signature uses the default tolerance for the appropriate floating point
 * operations.
 */
template<class ContainerE, class ContainerA, class BinaryOp>
::testing::AssertionResult IsVecSoftEquivImpl(const ContainerE& expected,
                                              const char*       expected_expr,
                                              const ContainerA& actual,
                                              const char*       actual_expr,
                                              BinaryOp          comp)
{
    using Traits_t = TCT<ContainerE, ContainerA>;
    using Failed_t = FailedValue<typename Traits_t::first_type,
                                 typename Traits_t::second_type>;
    std::vector<Failed_t>      failures;
    ::testing::AssertionResult result = IsRangeEqImpl(std::begin(expected),
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
            result << "Actual values: " << celeritas::repr(actual) << ";\n";
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

//---------------------------------------------------------------------------//
/*!
 * Print failure results.
 */
template<class T1, class T2>
std::string failure_msg(const char*                             expected_expr,
                        const char*                             actual_expr,
                        const std::vector<FailedValue<T1, T2>>& failures)
{
    using RT1 = celeritas::ReprTraits<T1>;
    using RT2 = celeritas::ReprTraits<T2>;
    using std::setw;

    // Calculate how many digits we need to space out
    int idig = num_digits(failures.back().index);
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
    for (const auto& f : failures)
    {
        os << setw(idig) << f.index << ' ' << setw(vdig);
        RT1::print_value(os, f.expected);
        os << ' ' << setw(vdig);
        RT2::print_value(os, f.actual);
        os << '\n';
    }
    return os.str();
}

//---------------------------------------------------------------------------//
/*!
 * Print failure results for floating point comparisons.
 */
template<class T1, class T2>
std::string float_failure_msg(const char* expected_expr,
                              const char* actual_expr,
                              const std::vector<FailedValue<T1, T2>>& failures,
                              double abs_thresh)
{
    using std::setprecision;
    using std::setw;

    // Calculate how many digits we need to space out the index
    int idig = num_digits(failures.back().index);
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
    for (const auto& f : failures)
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

//---------------------------------------------------------------------------//
/*!
 * Print expected values.
 */
template<class Container>
void print_expected(const Container& data, std::string label)
{
    using RT  = celeritas::ReprTraits<Container>;
    using VRT = celeritas::ReprTraits<typename RT::value_type>;

    std::cout << "static const ";
    label.insert(0, "expected_");
    VRT::print_type(std::cout, label.c_str());
    std::cout << "[] = ";

    std::ios orig_state(nullptr);
    orig_state.copyfmt(std::cout);
    VRT::init(std::cout);
    RT::print_value(std::cout, data);
    std::cout.copyfmt(orig_state);

    std::cout << ";\n";
}

//---------------------------------------------------------------------------//
/*!
 * Compare two containers.
 */
template<class ContainerE, class ContainerA>
::testing::AssertionResult IsVecEq(const char*       expected_expr,
                                   const char*       actual_expr,
                                   const ContainerE& expected,
                                   const ContainerA& actual)
{
    using Traits_t = TCT<ContainerE, ContainerA>;

    typename Traits_t::Failed_Vec_t failures;

    ::testing::AssertionResult result
        = IsRangeEqImpl(std::begin(expected),
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
            result << "Actual values: " << celeritas::repr(actual) << ";\n";
        }
        else
        {
            // Print indices that were different
            result << failure_msg(expected_expr, actual_expr, failures);
        }
    }

    return result;
}

//-------------------------------------------------------------------------//
/*!
 * Compare two containers using soft equivalence.
 */
template<class ContainerE, class ContainerA>
::testing::AssertionResult IsVecSoftEquiv(const char*       expected_expr,
                                          const char*       actual_expr,
                                          const ContainerE& expected,
                                          const ContainerA& actual)
{
    using Traits_t     = TCT<ContainerE, ContainerA>;
    using value_type_E = typename Traits_t::first_type;
    using value_type_A = typename Traits_t::second_type;

    typename Traits_t::Failed_Vec_t failures;

    static_assert(can_soft_equiv<value_type_E, value_type_A>(),
                  "Invalid types for soft equivalence");

    using ValueT = typename SoftPrecisionType<value_type_E, value_type_A>::type;
    using BinaryOp = celeritas::SoftEqual<ValueT>;

    // Construct with automatic or specified tolerances
    return IsVecSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp());
}

//-------------------------------------------------------------------------//
/*!
 * Compare two containers using soft equivalence.
 *
 * This signature uses the default tolerance for the appropriate floating point
 * operations.
 */
template<class ContainerE, class ContainerA>
::testing::AssertionResult IsVecSoftEquiv(const char* expected_expr,
                                          const char* actual_expr,
                                          const char*,
                                          const ContainerE& expected,
                                          const ContainerA& actual,
                                          double            rel)
{
    using Traits_t     = TCT<ContainerE, ContainerA>;
    using value_type_E = typename Traits_t::first_type;
    using value_type_A = typename Traits_t::second_type;

    static_assert(can_soft_equiv<value_type_E, value_type_A>(),
                  "Invalid types for soft equivalence");

    using ValueT = typename SoftPrecisionType<value_type_E, value_type_A>::type;
    using BinaryOp = celeritas::SoftEqual<ValueT>;

    // Construct with given tolerance
    return IsVecSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp(rel));
}

//-------------------------------------------------------------------------//
/*!
 * Compare two containers using soft equivalence.
 *
 * Used by \c EXPECT_VEC_CLOSE.
 */
template<class ContainerE, class ContainerA>
::testing::AssertionResult IsVecSoftEquiv(const char* expected_expr,
                                          const char* actual_expr,
                                          const char*,
                                          const char*,
                                          const ContainerE& expected,
                                          const ContainerA& actual,
                                          double            rel,
                                          double            abs)
{
    using Traits_t     = TCT<ContainerE, ContainerA>;
    using value_type_E = typename Traits_t::first_type;
    using value_type_A = typename Traits_t::second_type;

    static_assert(can_soft_equiv<value_type_E, value_type_A>(),
                  "Invalid types for soft equivalence");

    using ValueT = typename SoftPrecisionType<value_type_E, value_type_A>::type;
    using BinaryOp = celeritas::SoftEqual<ValueT>;

    // Construct with given tolerance
    return IsVecSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp(rel, abs));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas_test
