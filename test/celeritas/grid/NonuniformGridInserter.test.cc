//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridInserter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/NonuniformGridInserter.hh"

#include <array>

#include "corecel/OpaqueId.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class NonuniformGridInserterTest : public ::celeritas::test::Test
{
  protected:
    using GridIndexType = OpaqueId<struct NonuniformIndexTag_>;
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;
    using VecDbl = std::vector<double>;

    void SetUp() override { rng_.reset_count(); }

    NonuniformGridInserter<GridIndexType> make_inserter()
    {
        return NonuniformGridInserter<GridIndexType>(&scalars_, &grids_);
    }

    //! Construct an array of random, increasing data to test on
    VecDbl build_random_array(size_type count, real_type start)
    {
        UniformRealDistribution sample_uniform(0.5, 1.5);
        VecDbl xs(count);
        xs[0] = start;
        for (auto i : range<size_type>(1, xs.size()))
        {
            xs[i] = xs[i - 1] + sample_uniform(rng_);
        }
        return xs;
    }

    //! Check that an inserted grid has been constructed correctly
    void check_grid(GridIndexType id, VecDbl const& xs, VecDbl const& ys) const
    {
        ASSERT_EQ(xs.size(), ys.size());
        ASSERT_TRUE(id);
        ASSERT_LT(id.get(), grids_.size());

        NonuniformGridRecord const& grid = grids_[id];
        EXPECT_VEC_SOFT_EQ(xs, scalars_[grid.grid]);
        EXPECT_VEC_SOFT_EQ(ys, scalars_[grid.value]);
    }

    Collection<real_type, Ownership::value, MemSpace::host> scalars_;
    Collection<NonuniformGridRecord, Ownership::value, MemSpace::host, GridIndexType>
        grids_;

    RandomEngine rng_;
};

TEST_F(NonuniformGridInserterTest, simple)
{
    constexpr size_t count = 105;

    inp::Grid grid;
    grid.x = build_random_array(count, -100.0);
    grid.y = build_random_array(count, 300.0);

    auto insert = make_inserter();
    GridIndexType grid_index = insert(grid);

    ASSERT_EQ(1, grids_.size());
    ASSERT_EQ(2 * count, scalars_.size());

    check_grid(grid_index, grid.x, grid.y);
}

TEST_F(NonuniformGridInserterTest, many_no_repeats)
{
    constexpr size_t count = 58;
    auto insert = make_inserter();

    std::vector<GridIndexType> grid_ids;
    std::vector<inp::Grid> grids;

    size_t const num_grids = 20;
    for (size_t i = 0; i < num_grids; i++)
    {
        inp::Grid grid;
        grid.x = build_random_array(count, real_type{-100} * i);
        grid.y = build_random_array(count, real_type{300} * i);
        grids.push_back(grid);

        grid_ids.push_back(insert(grid));
    }

    ASSERT_EQ(num_grids, grids_.size());
    ASSERT_EQ(num_grids, grids.size());
    ASSERT_EQ(2 * count * num_grids, scalars_.size());

    for (size_t i = 0; i < num_grids; i++)
    {
        check_grid(grid_ids[i], grids[i].x, grids[i].y);
    }
}

TEST_F(NonuniformGridInserterTest, many_with_repeats)
{
    constexpr size_t count = 75;
    auto insert = make_inserter();

    std::vector<GridIndexType> grid_ids;
    std::vector<inp::Grid> grids;

    inp::Grid grid;
    grid.x = build_random_array(count, -100);

    size_t const num_grids = 20;
    for (size_t i = 0; i < num_grids; i++)
    {
        grid.y = build_random_array(count, real_type{300} * i);
        grids.push_back(grid);

        grid_ids.push_back(insert(grid));
    }

    ASSERT_EQ(num_grids, grids_.size());
    ASSERT_EQ(num_grids, grids.size());
    ASSERT_EQ(count * (num_grids + 1), scalars_.size());

    for (size_t i = 0; i < num_grids; i++)
    {
        check_grid(grid_ids[i], grids[i].x, grids[i].y);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
