//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsGridCalculator.test.cc
//---------------------------------------------------------------------------//
#include "physics/grid/PhysicsGridCalculator.hh"

#include <cmath>
#include "base/Interpolator.hh"
#include "base/Range.hh"
#include "celeritas_test.hh"

using celeritas::PhysicsGridCalculator;
using celeritas::UniformGridPointers;
using celeritas::XsGridPointers;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PhysicsGridCalculatorTest : public celeritas::Test
{
  protected:
    using Energy    = celeritas::real_type;
    using real_type = celeritas::real_type;

    void build(real_type emin, real_type emax, int count)
    {
        CELER_EXPECT(count >= 2);
        data.log_energy = UniformGridPointers::from_bounds(
            std::log(emin), std::log(emax), count);

        stored_xs.resize(count);

        // Interpolate xs grid: linear in bin, log in energy
        using celeritas::Interp;
        celeritas::Interpolator<Interp::linear, Interp::log, real_type> calc_xs(
            {0.0, emin}, {count - 1.0, emax});
        for (auto i : celeritas::range(stored_xs.size()))
        {
            stored_xs[i] = calc_xs(i);
        }
        data.value = celeritas::make_span(stored_xs);

        CELER_ENSURE(data);
        CELER_ENSURE(celeritas::soft_equal(emax, data.value.back()));
    }

    std::vector<real_type> stored_xs;
    XsGridPointers         data;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PhysicsGridCalculatorTest, simple)
{
    // Energy from 1 to 1e5 MeV with 5 grid points; XS should be the same
    // *No* magical 1/E scaling
    this->build(1.0, 1e5, 6);

    PhysicsGridCalculator calc(this->data);

    // Test on grid points
    EXPECT_SOFT_EQ(1.0, calc(Energy{1}));
    EXPECT_SOFT_EQ(1e2, calc(Energy{1e2}));
    EXPECT_SOFT_EQ(1e5 - 1e-6, calc(Energy{1e5 - 1e-6}));
    EXPECT_SOFT_EQ(1e5, calc(Energy{1e5}));

    // Test between grid points
    EXPECT_SOFT_EQ(5, calc(Energy{5}));

    // Test out-of-bounds
    EXPECT_SOFT_EQ(1.0, calc(Energy{0.0001}));
    EXPECT_SOFT_EQ(1e5, calc(Energy{1e7}));
}

TEST_F(PhysicsGridCalculatorTest, scaled)
{
    // TODO: test me and fix as needed
}
