//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ValidationUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>
#include <vector>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/io/ImportPhysicsVector.hh"
#include "celeritas/optical/MfpBuilder.hh"

#include "Test.hh"
#include "TestMacros.hh"

#include "detail/ValidationUtilsImpl.hh"

//! Equivalence macro for physics grids
#define EXPECT_GRID_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(::celeritas::testdetail::IsGridEq, expected, actual)

//! Equivalence macro for physics tables (vectors of grids)
#define EXPECT_TABLE_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(::celeritas::testdetail::IsTableEq, expected, actual)

namespace celeritas
{
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
 * Convenience class for accessing data built on the grid, and performing
 * sanity checks on bounds.
 */
class GridAccessor
{
  public:
    //!@{
    //! \name Type aliases
    using Grid = NonuniformGridRecord;
    using GridId = OpaqueId<Grid>;
    using ImportPhysicsTable = std::vector<ImportPhysicsVector>;
    using GridView = std::tuple<Span<real_type const>, Span<real_type const>>;

    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    //!@}

  public:
    // Construct accessor for underlying storage
    GridAccessor(Items<real_type>* reals, Items<Grid>* grids);

    // Retrieve a table of grid views built on the storage
    std::vector<GridView> operator()(ItemRange<Grid> grid_ids) const;

    // Retrieve a span of reals built on the storage
    Span<real_type const>
    operator()(ItemRange<real_type> const& real_ids) const;

    // Construct an MFP builder with the underlying collections
    MfpBuilder create_mfp_builder();

  private:
    Items<real_type>* reals_;
    Items<Grid>* grids_;
};

//---------------------------------------------------------------------------//
/*!
 * A \c GridAccessor that stores its own collections.
 */
class OwningGridAccessor : public GridAccessor
{
  public:
    // Construct with internal collections
    OwningGridAccessor();

  private:
    Items<real_type> reals_;
    Items<Grid> grids_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
