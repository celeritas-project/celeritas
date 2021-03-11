//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TwodGridCalculator.test.cc
//---------------------------------------------------------------------------//
#include "physics/grid/TwodGridCalculator.hh"

#include "base/Collection.hh"
#include "base/CollectionBuilder.hh"
#include "celeritas_test.hh"

using celeritas::Ownership;
using celeritas::range;
using celeritas::real_type;
using celeritas::TwodGridCalculator;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class TwodGridCalculatorTest : public celeritas::Test
{
  protected:
    template<Ownership W>
    using RealData
        = celeritas::Collection<real_type, W, celeritas::MemSpace::host>;

    void SetUp() override
    {
        xgrid_ = {-1, 0, 1, 3};
        ygrid_ = {0, 0.5, 1.5, 3.5};

        auto build   = celeritas::make_builder(&values_);
        grid_data_.x = build.insert_back(xgrid_.begin(), xgrid_.end()); // X
        grid_data_.y = build.insert_back(ygrid_.begin(), ygrid_.end()); // Y

        const auto             nx = xgrid_.size();
        const auto             ny = ygrid_.size();
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

    // Bilinear function of (x, y) should exactly reproduce with interpolation
    real_type calc_expected(real_type x, real_type y) const
    {
        return 1 + x + 2 * y - 0.5 * x * y;
    }

    std::vector<real_type> xgrid_;
    std::vector<real_type> ygrid_;

    celeritas::TwodGridData              grid_data_;
    RealData<Ownership::value>           values_;
    RealData<Ownership::const_reference> ref_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TwodGridCalculatorTest, all)
{
    TwodGridCalculator interpolate(grid_data_, ref_);

    // Exact point
    EXPECT_SOFT_EQ(1.0, interpolate({0.0, 0.0}));

    // Outer extrema
    const real_type eps = 1e-6;
    EXPECT_SOFT_EQ(calc_expected(-1, 0), interpolate({-1, 0}));
    EXPECT_SOFT_EQ(calc_expected(3 - eps, 3.5 - eps),
                   interpolate({3 - eps, 3.5 - eps}));

    // Interior points
    for (real_type x : {-1.0, -0.5, -0.9, 2.25})
    {
        for (real_type y : {0.0, 0.4, 1.6, 3.25})
        {
            EXPECT_SOFT_EQ(calc_expected(x, y), interpolate({x, y}));
        }
    }
}
