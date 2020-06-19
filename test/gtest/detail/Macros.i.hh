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

namespace celeritas
{
namespace detail
{
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
} // namespace detail
} // namespace celeritas
