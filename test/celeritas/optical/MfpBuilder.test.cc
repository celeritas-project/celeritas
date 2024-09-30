//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MfpBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/detail/MfpBuilder.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//

using namespace ::celeritas::test;

// Convert vector of doubles to vector of real_type
std::vector<real_type> convert_to_reals(std::vector<double> const& xs)
{
    std::vector<real_type> reals;
    reals.reserve(xs.size());
    for (double x : xs)
    {
        reals.push_back(static_cast<real_type>(x));
    }
    return reals;
}

class MfpBuilderTest : public ::celeritas::test::Test
{
  protected:
    using Grid = GenericGridRecord;
    using GridId = OpaqueId<Grid>;

    template <class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;

    void SetUp() override {}

    // Mock grid data
    static std::vector<ImportPhysicsVector> const& expected_mfps()
    {
        constexpr auto linear = ImportPhysicsVectorType::linear;
        static std::vector<ImportPhysicsVector> mfps = {
            {linear, {1e-3, 1e2}, {5.7, 10.3}},
            {linear, {1e-3, 1e1, 1e2}, {1.9, 8.1, 10.8}},
            {linear, {1e-3, 1e2}, {3.2, 9.1}},
        };

        return mfps;
    }


    // Signature to mimic optical Model call
    void build_mfps(detail::MfpBuilder& build) const
    {
        for (auto const& mfp : this->expected_mfps())
        {
            build(mfp);
        }
    }

    void check_table(ItemRange<Grid> const& table) const
    {
        // Each MFP has a grid
        EXPECT_EQ(this->expected_mfps().size(), table.size());

        for (auto grid_id : table)
        {
            // Grid IDs should be valid
            ASSERT_LT(grid_id, grids.size());
            
            // Grid should be valid
            Grid const& grid = grids[grid_id];
            ASSERT_TRUE(grid);

            // Grid ranges should be valid
            ASSERT_LT(grid.grid.back(), reals.size());
            ASSERT_LT(grid.value.back(), reals.size());

            // Convert imported data to real_type for comparison
            auto const& expected_mfp = this->expected_mfps()[grid_id.get()];
            std::vector<real_type> expected_grid = convert_to_reals(expected_mfp.x);
            std::vector<real_type> expected_value = convert_to_reals(expected_mfp.y);

            // Built grid data should match expected grid data
            EXPECT_VEC_EQ(expected_grid, reals[grid.grid]);
            EXPECT_VEC_EQ(expected_value, reals[grid.value]);
        }
    }

    Items<real_type> reals;
    Items<Grid> grids;
};


// Check MFP tables are built with correct structure from imported data
TEST_F(MfpBuilderTest, construct_table)
{
    detail::MfpBuilder builder(&reals, &grids);

    this->build_mfps(builder);

    auto table = builder.grid_ids();
    this->check_table(table);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
