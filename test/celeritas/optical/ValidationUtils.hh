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

namespace celeritas
{

namespace testdetail
{
template<>
::testing::AssertionResult
IsVecEq<ImportPhysicsVector, ImportPhysicsVector>(char const*,
                                                  char const*,
                                                  ImportPhysicsVector const&,
                                                  ImportPhysicsVector const&);
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

    //! Whether to use soft or exact equivalence
    enum Softness
    {
        Soft,
        Exact
    };

  public:
    // Construct validator for underlying storage
    GridValidator(Items<real_type>* reals, Items<Grid>* grids);

    // Check the imported data is built under the given grid ID range
    void check_built_table(ImportPhysicsTable const& table,
                           ItemRange<Grid> grid_ids,
                           Softness soft);

    // Check the imported data is built under the given grid ID
    void check_built_grid(ImportPhysicsVector const& expected,
                          GridId grid_id,
                          Softness soft);

    //! Check the imported data is built in the given data range
    template<class T>
    void check_built_vector(T const& t,
                            ItemRange<real_type> const& real_ids,
                            Softness soft)
    {
        ASSERT_LT(real_ids.front(), real_ids.back());
        ASSERT_LT(real_ids.back(), reals_->size());

        switch (soft)
        {
            case Soft:
                EXPECT_VEC_SOFT_EQ(t, (*reals_)[real_ids]);
                break;
            case Exact:
                EXPECT_VEC_EQ(t, (*reals_)[real_ids]);
                break;
        }
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
