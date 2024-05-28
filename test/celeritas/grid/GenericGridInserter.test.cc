//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridInserter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/GenericGridInserter.hh"

#include <array>

#include "corecel/OpaqueId.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class GenericGridInserterTest : public ::celeritas::test::Test
{
  protected:
    using GridIndexType = OpaqueId<struct GenericIndexTag_>;
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;

    void SetUp() override { rng_.reset_count(); }

    GenericGridInserter<GridIndexType> make_inserter()
    {
        return GenericGridInserter<GridIndexType>(&scalars_, &grids_);
    }

    //! Construct an array of random, increasing data to test on
    template<size_t N>
    std::array<real_type, N> build_random_array(real_type start)
    {
        UniformRealDistribution dist(0.5, 1.5);
        std::array<real_type, N> xs;
        xs[0] = start;
        for (size_t i = 1; i < N; i++)
        {
            xs[i] = xs[i - 1] + dist(rng_);
        }
        return xs;
    }

    //! Check that an inserted grid has been constructed correctly
    template<size_t N>
    void check_grid(GridIndexType id,
                    std::array<real_type, N> const& xs,
                    std::array<real_type, N> const& ys) const
    {
        ASSERT_TRUE(id);
        ASSERT_LT(id.get(), grids_.size());

        GenericGridData const& grid = grids_[id];
        ASSERT_EQ(N, grid.grid.size());
        ASSERT_EQ(N, grid.value.size());

        EXPECT_VEC_EQ(xs, scalars_[grid.grid]);
        EXPECT_VEC_EQ(ys, scalars_[grid.value]);
    }

    Collection<real_type, Ownership::value, MemSpace::host> scalars_;
    Collection<GenericGridData, Ownership::value, MemSpace::host, GridIndexType>
        grids_;

    RandomEngine rng_;
};

TEST_F(GenericGridInserterTest, simple)
{
    constexpr size_t count = 105;
    auto const xs = build_random_array<count>(-100.0);
    auto const ys = build_random_array<count>(300.0);

    auto inserter = make_inserter();

    GridIndexType grid_index = inserter(make_span(xs), make_span(ys));

    ASSERT_EQ(1, grids_.size());
    ASSERT_EQ(2 * count, scalars_.size());

    check_grid<count>(grid_index, xs, ys);
}

TEST_F(GenericGridInserterTest, many_no_repeats)
{
    constexpr size_t count = 58;
    auto inserter = make_inserter();

    std::vector<GridIndexType> grid_ids;
    std::vector<std::array<real_type, count>> raw_xs, raw_ys;

    size_t const num_grids = 20;
    for (size_t i = 0; i < num_grids; i++)
    {
        raw_xs.push_back(build_random_array<count>(-100.0 * i));
        raw_ys.push_back(build_random_array<count>(300.0 * i));

        auto const& xs = raw_xs.back();
        auto const& ys = raw_ys.back();

        grid_ids.push_back(inserter(make_span(xs), make_span(ys)));
    }

    ASSERT_EQ(num_grids, grids_.size());
    ASSERT_EQ(num_grids, raw_xs.size());
    ASSERT_EQ(num_grids, raw_ys.size());
    ASSERT_EQ(2 * count * num_grids, scalars_.size());

    for (size_t i = 0; i < num_grids; i++)
    {
        check_grid(grid_ids[i], raw_xs[i], raw_ys[i]);
    }
}

TEST_F(GenericGridInserterTest, many_with_repeats)
{
    constexpr size_t count = 75;
    auto inserter = make_inserter();

    std::vector<GridIndexType> grid_ids;
    std::array<real_type, count> xs = build_random_array<count>(-100.0);
    std::vector<std::array<real_type, count>> raw_ys;

    size_t const num_grids = 20;
    for (size_t i = 0; i < num_grids; i++)
    {
        raw_ys.push_back(build_random_array<count>(300.0 * i));

        auto const& ys = raw_ys.back();

        grid_ids.push_back(inserter(make_span(xs), make_span(ys)));
    }

    ASSERT_EQ(num_grids, grids_.size());
    ASSERT_EQ(num_grids, raw_ys.size());
    ASSERT_EQ(count * (num_grids + 1), scalars_.size());

    for (size_t i = 0; i < num_grids; i++)
    {
        check_grid(grid_ids[i], xs, raw_ys[i]);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
