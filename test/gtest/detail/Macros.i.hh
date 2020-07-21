//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Macros.i.hh
//---------------------------------------------------------------------------//

#include <type_traits>
#include <gtest/gtest.h>

#include "base/SoftEqual.hh"
#include "PrintableValueTraits.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// SOFT EQUIVALENCE
//---------------------------------------------------------------------------//
//! Whether soft equivalence can be performed on the given types
template<typename T1, typename T2>
constexpr bool can_soft_equiv()
{
    return (std::is_floating_point<T1>::value
            || std::is_floating_point<T2>::value)
           && std::is_convertible<T1, T2>::value;
}

//---------------------------------------------------------------------------//
//! Compare a range of values
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

    result << "Value of: " << actual_expr
           << "\n"
              "  Actual: "
           << actual
           << "\n"
              "Expected: "
           << expected_expr
           << "\n"
              "Which is: "
           << expected << '\n';

    SoftZero<value_type> is_soft_zero(comp.abs());
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
 * \brief Predicate for relative error soft equiavelence
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
    using BinaryOp = celeritas::SoftEqual<Value_E, Value_A>;

    return IsSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Predicate for relative error soft equiavelence
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
    using BinaryOp = celeritas::SoftEqual<Value_E, Value_A>;

    return IsSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp(rel));
}

//---------------------------------------------------------------------------//
// CONTAINER EQUIVALENCE
//---------------------------------------------------------------------------//
//! A single index/expected/actual value
template<typename T1, typename T2>
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
    using first_type =
        typename std::remove_const<typename ContTraits<C1>::value_type>::type;
    using second_type =
        typename std::remove_const<typename ContTraits<C2>::value_type>::type;

    using common_type =
        typename std::common_type<first_type, second_type>::type;

    using Failed_Value_t = FailedValue<first_type, second_type>;
    using Failed_Vec_t   = std::vector<Failed_Value_t>;
};

// Failed value iterator traits
template<typename Iter1, typename Iter2>
struct FVIT
{
    using first_type = typename std::remove_const<
        typename std::iterator_traits<Iter1>::value_type>::type;
    using second_type = typename std::remove_const<
        typename std::iterator_traits<Iter2>::value_type>::type;

    using type  = FailedValue<first_type, second_type>;
    using Vec_t = std::vector<type>;
};

//---------------------------------------------------------------------------//
//! Compare a range of values
template<typename Iter1, typename Iter2, class BinaryOp>
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

        failure << " Size of: " << actual_expr
                << "\n"
                   "  Actual: "
                << actual_size
                << "\n"
                   "Expected: "
                << expected_expr
                << ".size()\n"
                   "Which is: "
                << expected_size << '\n';
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
    result << "Values in: " << actual_expr
           << "\n"
              " Expected: "
           << expected_expr
           << "\n"
              ""
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
 * \brief Custom vector comparison with soft equiavelence
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
    using Failed_t = FailedValue<typename BinaryOp::first_argument_type,
                                 typename BinaryOp::second_argument_type>;
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
            result << "Actual values: " << to_string(actual) << ";\n";
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
//! Print failure results
template<class T1, class T2>
std::string failure_msg(const char*                             expected_expr,
                        const char*                             actual_expr,
                        const std::vector<FailedValue<T1, T2>>& failures)
{
    using PVT1 = PrintableValueTraits<T1>;
    using PVT2 = PrintableValueTraits<T2>;
    using std::setprecision;
    using std::setw;

    // Calculate how many digits we need to space out
    unsigned int idig = num_digits(failures.back().index);
    unsigned int vdig = 16;

    // Construct our own stringstream because google test ignores setw
    std::ostringstream os;
    PVT2::init(os);
    PVT1::init(os);

    // Print column headers (unless expected/actual is too long)
    os << setw(idig) << 'i' << ' ' << setw(vdig)
       << trunc_string(vdig, expected_expr, "EXPECTED") << ' ' << setw(vdig)
       << trunc_string(vdig, actual_expr, "ACTUAL") << '\n';

    // Loop through failed indices and print values
    for (const auto& f : failures)
    {
        os << setw(idig) << f.index << ' ' << setw(vdig);
        PVT1::print(os, f.expected);
        os << ' ' << setw(vdig);
        PVT2::print(os, f.actual);
        os << '\n';
    }
    return os.str();
}

//---------------------------------------------------------------------------//
//! Print failure results for floating point comparisons
template<class T1, class T2>
std::string float_failure_msg(const char* expected_expr,
                              const char* actual_expr,
                              const std::vector<FailedValue<T1, T2>>& failures,
                              double abs_thresh)
{
    using std::setprecision;
    using std::setw;

    // Calculate how many digits we need to space out the index
    unsigned int idig = num_digits(failures.back().index);
    unsigned int vdig = std::max(std::numeric_limits<T1>::digits10,
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
 * \brief Print expected values.
 */
template<class Container>
void print_expected(const Container& data, const char* label)
{
    using value_type = typename ContTraits<Container>::value_type;
    using PVT        = PrintableValueTraits<value_type>;
    using std::cout;
    using std::endl;

    cout << "const " << PVT::name() << " expected_" << label
         << "[] = " << to_string(data) << ";\n";
}

//---------------------------------------------------------------------------//
/*!
 * \brief Vector comparison.
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
            result << "Actual values: " << to_string(actual) << ";\n";
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
 * \brief Custom vector comparison with soft equivalence
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

    using BinaryOp = celeritas::SoftEqual<value_type_E, value_type_A>;

    // Construct with automatic or specified tolerances
    return IsVecSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp());
}

//-------------------------------------------------------------------------//
/*!
 * \brief Custom vector comparison with soft equiavelence
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

    using BinaryOp = celeritas::SoftEqual<value_type_E, value_type_A>;

    // Construct with given tolerance
    return IsVecSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp(rel));
}

//-------------------------------------------------------------------------//
/*!
 * \brief Custom vector comparison with soft equiavelence
 *
 * Used by EXPECT_VEC_CLOSE
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

    using BinaryOp = celeritas::SoftEqual<value_type_E, value_type_A>;

    // Construct with given tolerance
    return IsVecSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp(rel, abs));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
