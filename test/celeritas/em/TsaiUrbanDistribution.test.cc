//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/TsaiUrbanDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/distribution/TsaiUrbanDistribution.hh"

#include <random>

#include "celeritas/Constants.hh"
#include "celeritas/Units.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(TsaiUrbanDistributionTest, bin)
{
    using namespace constants;
    using namespace units;

    MevMass electron_mass = MevMass{0.5109989461};
    std::vector<real_type> angles;
    std::mt19937 rng;

    // Loop over various electron energies(converted to MevEnergy)
    for (real_type inc_e : {0.1, 1.0, 10.0, 50.0, 100.0})
    {
        TsaiUrbanDistribution sample_angle(MevEnergy{inc_e}, electron_mass);
        real_type angle = sample_angle(rng);
        angles.push_back(angle);
    }

    real_type const expected_angles[] = {0.527559321801249,
                                         0.882599596355283,
                                         0.999055310334017,
                                         0.999998183489194,
                                         0.999978220994207};
    EXPECT_VEC_SOFT_EQ(expected_angles, angles);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
