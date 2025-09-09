//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/ValidationUtilsImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "corecel/cont/Array.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
template<class UnitType, class... Args>
Array<real_type, sizeof...(Args)> constexpr native_array_from(Args const&...);
}  // namespace test
}  // namespace optical

namespace testdetail
{
//---------------------------------------------------------------------------//
/*!
 * Type traits for physics grids.
 *
 * Allows duck-typing for comparisons of physics grids that might be
 * stored in different forms and floating point precisions.
 */
template<class GridType>
struct PhysicsGridTraits;

//---------------------------------------------------------------------------//
//! Specialization for \c inp::Grid.
template<>
struct PhysicsGridTraits<inp::Grid>
{
    using grid_type = inp::Grid;

    static constexpr std::vector<double> const& grid(grid_type const& v)
    {
        return v.x;
    }
    static constexpr std::vector<double> const& value(grid_type const& v)
    {
        return v.y;
    }
};

//---------------------------------------------------------------------------//
//! Specialization for a tuple of containers
template<class T>
struct PhysicsGridTraits<std::tuple<T, T>>
{
    using grid_type = std::tuple<T, T>;
    static constexpr T const& grid(grid_type const& v)
    {
        return std::get<0>(v);
    }
    static constexpr T const& value(grid_type const& v)
    {
        return std::get<1>(v);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Compare to physics grids with exact equivalence.
 */
template<class GridTypeE, class GridTypeA>
::testing::AssertionResult IsGridEq(char const* expected_expr,
                                    char const* actual_expr,
                                    GridTypeE const& expected,
                                    GridTypeA const& actual)
{
    using EGT = PhysicsGridTraits<GridTypeE>;
    using AGT = PhysicsGridTraits<GridTypeA>;

    // Compare grids (x values)
    auto x_result = IsVecEq(
        expected_expr, actual_expr, EGT::grid(expected), AGT::grid(actual));

    // Compare values (y values)
    auto y_result = IsVecEq(
        expected_expr, actual_expr, EGT::value(expected), AGT::value(actual));

    // Require both x and y successful to pass
    ::testing::AssertionResult result(x_result && y_result);
    if (!x_result)
    {
        result << "x values:\n" << x_result.message();
    }
    if (!y_result)
    {
        result << "y values:\n" << y_result.message();
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Compare physics tables with exact equivalence.
 */
template<class GridTypeE, class GridTypeA>
::testing::AssertionResult IsTableEq(char const* expected_expr,
                                     char const* actual_expr,
                                     std::vector<GridTypeE> const& expected,
                                     std::vector<GridTypeA> const& actual)
{
    // Check tables are equal size
    if (expected.size() != actual.size())
    {
        ::testing::AssertionResult failure = ::testing::AssertionFailure();

        failure << "   Size of: " << actual_expr
                << "\n  Actual: " << actual.size()
                << "\nExpected: " << expected_expr
                << ".size()\nWhich is: " << expected.size() << "\n";
        return failure;
    }

    ::testing::AssertionResult result = ::testing::AssertionSuccess();

    for (auto i : range(expected.size()))
    {
        // Extra formatting for table index
        std::string index_expr = "[" + std::to_string(i) + "]";
        std::string expected_expr_i = expected_expr + index_expr;
        std::string actual_expr_i = actual_expr + index_expr;

        // Compare table elements as physics grids
        auto grid_result = IsGridEq(expected_expr_i.c_str(),
                                    actual_expr_i.c_str(),
                                    expected[i],
                                    actual[i]);

        if (!grid_result)
        {
            // First failure needs to update result
            if (result)
            {
                result = ::testing::AssertionFailure();
            }

            // Append failure message
            result << grid_result.message();
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
