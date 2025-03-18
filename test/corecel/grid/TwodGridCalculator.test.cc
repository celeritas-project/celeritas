//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/TwodGridCalculator.test.cc
//---------------------------------------------------------------------------//

#include "corecel/grid/TwodGridCalculator.hh"

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/FindInterp.hh"
#include "corecel/grid/UniformGrid.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class TwodGridCalculatorTest : public Test
{
  protected:
    template<Ownership W>
    using RealData = Collection<real_type, W, MemSpace::host>;

    void SetUp() override
    {
        xgrid_ = {-1, 0, 1, 3};
        ygrid_ = {0, 0.5, 1.5, 3.5};

        auto build = make_builder(&values_);
        grid_data_.x = build.insert_back(xgrid_.begin(), xgrid_.end());  // X
        grid_data_.y = build.insert_back(ygrid_.begin(), ygrid_.end());  // Y

        auto const nx = xgrid_.size();
        auto const ny = ygrid_.size();
        std::vector<real_type> values(nx * ny);
        for (auto i : range(xgrid_.size()))
        {
            for (auto j : range(ygrid_.size()))
            {
                values[i * ny + j] = this->calc_expected(xgrid_[i], ygrid_[j]);
            }
        }
        grid_data_.values = build.insert_back(values.begin(), values.end());
        EXPECT_EQ(grid_data_.values.front(), grid_data_.at(0, 0));

        CELER_ENSURE(grid_data_);
        ref_ = values_;
    }

    // Bilinear function of (x, y) should exactly reproduce with
    // interpolation
    real_type calc_expected(real_type x, real_type y) const
    {
        return 1 + x + 2 * y - 0.5 * x * y;
    }

    std::vector<real_type> xgrid_;
    std::vector<real_type> ygrid_;

    TwodGridData grid_data_;
    RealData<Ownership::value> values_;
    RealData<Ownership::const_reference> ref_;
};

TEST_F(TwodGridCalculatorTest, whole_grid)
{
    TwodGridCalculator interpolate(grid_data_, ref_);

    // Exact point
    EXPECT_SOFT_EQ(1.0, interpolate({0.0, 0.0}));

    // Outer extrema
    real_type const eps = 1e-6;
    EXPECT_SOFT_EQ(calc_expected(-1, 0), interpolate({-1, 0}));
    EXPECT_SOFT_EQ(calc_expected(3 - eps, real_type(3.5) - eps),
                   interpolate({3 - eps, real_type(3.5) - eps}));

    // Interior points
    for (real_type x : {-1.0, -0.5, -0.9, 2.25})
    {
        for (real_type y : {0.0, 0.4, 1.6, 3.25})
        {
            EXPECT_SOFT_EQ(calc_expected(x, y), interpolate({x, y}));
        }
    }
}

TEST_F(TwodGridCalculatorTest, subgrid)
{
    std::vector<size_type> lower_idx;
    std::vector<real_type> frac;

    for (real_type x : {-1., .5, 2.99})
    {
        auto interpolate = TwodGridCalculator(grid_data_, ref_)(x);
        lower_idx.push_back(interpolate.x_index());
        frac.push_back(interpolate.x_fraction());
        for (real_type y : {0.0, 0.4, 1.6, 3.25})
        {
            EXPECT_SOFT_EQ(calc_expected(x, y), interpolate(y));
        }
        for (size_type i : range(ygrid_.size()))
        {
            EXPECT_SOFT_EQ(calc_expected(x, ygrid_[i]), interpolate[i]);
        }
    }

    unsigned int const expected_lower_idx[] = {0u, 1u, 2u};
    double const expected_frac[] = {0, 0.5, 0.995};

    EXPECT_VEC_EQ(expected_lower_idx, lower_idx);
    EXPECT_VEC_SOFT_EQ(expected_frac, frac);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
