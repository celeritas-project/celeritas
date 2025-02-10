//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridInserter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/NonuniformGridInserter.hh"

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

class NonuniformGridInserterTest : public ::celeritas::test::Test
{
  protected:
    using GridIndexType = OpaqueId<struct NonuniformIndexTag_>;
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;
    using VecReal = std::vector<real_type>;

    void SetUp() override { rng_.reset_count(); }

    NonuniformGridInserter<GridIndexType> make_inserter()
    {
        return NonuniformGridInserter<GridIndexType>(&scalars_, &grids_);
    }

    //! Construct an array of random, increasing data to test on
    VecReal build_random_array(size_type count, real_type start)
    {
        UniformRealDistribution sample_uniform(0.5, 1.5);
        VecReal xs(count);
        xs[0] = start;
        for (auto i : range<size_type>(1, xs.size()))
        {
            xs[i] = xs[i - 1] + sample_uniform(rng_);
        }
        return xs;
    }

    //! Check that an inserted grid has been constructed correctly
    void check_grid(GridIndexType id,
                    std::vector<real_type> const& xs,
                    std::vector<real_type> const& ys) const
    {
        ASSERT_EQ(xs.size(), ys.size());
        ASSERT_TRUE(id);
        ASSERT_LT(id.get(), grids_.size());

        NonuniformGridRecord const& grid = grids_[id];
        EXPECT_VEC_EQ(xs, scalars_[grid.grid]);
        EXPECT_VEC_EQ(ys, scalars_[grid.value]);
    }

    Collection<real_type, Ownership::value, MemSpace::host> scalars_;
    Collection<NonuniformGridRecord, Ownership::value, MemSpace::host, GridIndexType>
        grids_;

    RandomEngine rng_;
};

TEST_F(NonuniformGridInserterTest, simple)
{
    constexpr size_t count = 105;
    auto const xs = build_random_array(count, -100.0);
    auto const ys = build_random_array(count, 300.0);

    auto inserter = make_inserter();

    GridIndexType grid_index = inserter(make_span(xs), make_span(ys));

    ASSERT_EQ(1, grids_.size());
    ASSERT_EQ(2 * count, scalars_.size());

    check_grid(grid_index, xs, ys);
}

TEST_F(NonuniformGridInserterTest, many_no_repeats)
{
    constexpr size_t count = 58;
    auto inserter = make_inserter();

    std::vector<GridIndexType> grid_ids;
    std::vector<VecReal> raw_xs, raw_ys;

    size_t const num_grids = 20;
    for (size_t i = 0; i < num_grids; i++)
    {
        raw_xs.push_back(build_random_array(count, real_type{-100} * i));
        raw_ys.push_back(build_random_array(count, real_type{300} * i));

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

TEST_F(NonuniformGridInserterTest, many_with_repeats)
{
    constexpr size_t count = 75;
    auto inserter = make_inserter();

    std::vector<GridIndexType> grid_ids;
    VecReal xs = build_random_array(count, -100);
    std::vector<VecReal> raw_ys;

    size_t const num_grids = 20;
    for (size_t i = 0; i < num_grids; i++)
    {
        raw_ys.push_back(build_random_array(count, real_type{300} * i));

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
