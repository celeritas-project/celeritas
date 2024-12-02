//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ValidationUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>
#include <type_traits>
#include <vector>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/io/ImportPhysicsVector.hh"
#include "celeritas/optical/MfpBuilder.hh"

#include "Test.hh"
#include "TestMacros.hh"

//! Equivalence macro for physics grids
#define EXPECT_GRID_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(::celeritas::testdetail::IsGridEq, expected, actual)

//! Equivalence macro for physics tables (vectors of grids)
#define EXPECT_TABLE_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(::celeritas::testdetail::IsTableEq, expected, actual)

namespace celeritas
{
namespace testdetail
{
//---------------------------------------------------------------------------//
/*!
 * Type traits for physics grids.
 *
 * Allows duck-typing to allow comparisons of physics grids that might be
 * stored as different objects.
 */
template<class GridType>
struct PhysicsGridTraits;

//---------------------------------------------------------------------------//
//! Specialization for \c ImportPhysicsVector
template<>
struct PhysicsGridTraits<ImportPhysicsVector>
{
    using grid_type = ImportPhysicsVector;

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

    auto x_result = IsVecEq(
        expected_expr, actual_expr, EGT::grid(expected), AGT::grid(actual));
    auto y_result = IsVecEq(
        expected_expr, actual_expr, EGT::value(expected), AGT::value(actual));

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
    if (expected.size() != actual.size())
    {
        ::testing::AssertionResult failure = ::testing::AssertionFailure();

        failure << " Size of: " << actual_expr
                << "\n  Actual: " << actual.size()
                << "\nExpected: " << expected_expr
                << ".size()\nWhich is: " << expected.size() << "\n";
        return failure;
    }

    ::testing::AssertionResult result = ::testing::AssertionSuccess();

    for (auto i : range(expected.size()))
    {
        std::string index_expr = "[" + std::to_string(i) + "]";
        std::string expected_expr_i = expected_expr + index_expr;
        std::string actual_expr_i = actual_expr + index_expr;

        auto grid_result = IsGridEq(expected_expr_i.c_str(),
                                    actual_expr_i.c_str(),
                                    expected[i],
                                    actual[i]);

        if (!grid_result)
        {
            if (result)
            {
                result = ::testing::AssertionFailure();
            }

            result << grid_result.message();
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace testdetail

namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
/*!
 * Helper function to annotate units of a hard-coded test data array.
 *
 * Converts the arguments supplied in units \c UnitType to native units.
 */
template<class UnitType, class... Args>
Array<real_type, sizeof...(Args)> constexpr native_array_from(Args const&... args)
{
    return Array<real_type, sizeof...(Args)>{
        native_value_from(UnitType(args))...};
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to annotate units of hard-coded test data.
 *
 * Same as \c native_array_from, except returns a vector.
 */
template<class UnitType, class... Args>
std::vector<real_type> native_vector_from(Args const&... args)
{
    return std::vector<real_type>{native_value_from(UnitType(args))...};
}

//---------------------------------------------------------------------------//
/*!
 * Perform consistency checks on grids built in \c Collections.
 */
class GridValidator
{
  public:
    //!@{
    //! \name Type aliases
    using Grid = GenericGridRecord;
    using GridId = OpaqueId<Grid>;
    using ImportPhysicsTable = std::vector<ImportPhysicsVector>;

    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    //!@}

  public:
    // Construct validator for underlying storage
    GridValidator(Items<real_type>* reals, Items<Grid>* grids);

    std::vector<std::tuple<Span<real_type const>, Span<real_type const>>>
    operator()(ItemRange<Grid> grid_ids) const
    {
        std::vector<std::tuple<Span<real_type const>, Span<real_type const>>> grids;
        grids.reserve(grid_ids.size());

        for (GridId grid_id : grid_ids)
        {
            CELER_EXPECT(grid_id < grids_->size());
            auto const& grid = (*grids_)[grid_id];
            CELER_EXPECT(grid);
            grids.push_back(
                std::make_tuple((*this)(grid.grid), (*this)(grid.value)));
        }

        return grids;
    }

    Span<real_type const> operator()(ItemRange<real_type> const& real_ids) const
    {
        CELER_EXPECT(real_ids.front() < real_ids.back());
        CELER_EXPECT(real_ids.back() < reals_->size());
        return (*reals_)[real_ids];
    }

    // Construct an MFP builder with the underlying collections
    MfpBuilder create_mfp_builder();

  private:
    Items<real_type>* reals_;
    Items<Grid>* grids_;
};

//---------------------------------------------------------------------------//
/*!
 * A \c GridValidator that stores its own collections.
 */
class OwningGridValidator : public GridValidator
{
  public:
    // Construct with internal collections
    OwningGridValidator();

  private:
    Items<real_type> reals_;
    Items<Grid> grids_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
