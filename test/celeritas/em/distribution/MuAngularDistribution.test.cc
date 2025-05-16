//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/MuAngularDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/distribution/MuAngularDistribution.hh"

#include "corecel/random/HistogramSampler.hh"
#include "celeritas/Quantities.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(MuAngularDistributionTest, costheta_dist)
{
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;

    constexpr Mass muon_mass{105.6583745};
    constexpr size_type num_samples = 1000;

    std::vector<SampledHistogram> actual;

    // Bin cos theta
    HistogramSampler calc_histogram(8, {-1, 1}, num_samples);

    for (real_type inc_e : {0.1, 1.0, 1e2, 1e3, 1e6})
    {
        for (real_type eps : {0.001, 0.01, 0.1})
        {
            MuAngularDistribution sample_mu{
                Energy{inc_e}, muon_mass, Energy{eps * inc_e}};
            actual.push_back(calc_histogram(sample_mu));
        }
    }

    // PRINT_EXPECTED(actual);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        SampledHistogram const expected[] = {
            {{0, 0, 0, 0, 0.484, 0.604, 0.96, 1.952}, 2},
            {{0, 0, 0, 0, 0.428, 0.624, 1.04, 1.908}, 2},
            {{0, 0, 0, 0, 0.5, 0.684, 0.908, 1.908}, 2},
            {{0, 0, 0, 0, 0.42, 0.58, 1.048, 1.952}, 2},
            {{0, 0, 0, 0, 0.404, 0.568, 1.088, 1.94}, 2},
            {{0, 0, 0, 0, 0.52, 0.528, 1.116, 1.836}, 2},
            {{0, 0, 0, 0, 0.156, 0.2, 0.608, 3.036}, 2},
            {{0, 0, 0, 0, 0.132, 0.244, 0.6, 3.024}, 2},
            {{0, 0, 0, 0, 0.176, 0.244, 0.668, 2.912}, 2},
            {{0, 0, 0, 0, 0.012, 0, 0.028, 3.96}, 2},
            {{0, 0, 0, 0, 0.004, 0.004, 0.032, 3.96}, 2},
            {{0, 0, 0, 0, 0.008, 0.004, 0.02, 3.968}, 2},
            {{0, 0, 0, 0, 0, 0, 0, 4}, 2},
            {{0, 0, 0, 0, 0, 0, 0, 4}, 2},
            {{0, 0, 0, 0, 0, 0, 0, 4}, 2},
        };
        EXPECT_REF_EQ(expected, actual);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
