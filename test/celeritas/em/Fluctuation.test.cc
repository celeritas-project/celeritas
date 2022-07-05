//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/Fluctuation.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/MockTestBase.hh"
#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/distribution/EnergyLossDeltaDistribution.hh"
#include "celeritas/em/distribution/EnergyLossGaussianDistribution.hh"
#include "celeritas/em/distribution/EnergyLossHelper.hh"
#include "celeritas/em/distribution/EnergyLossUrbanDistribution.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class FluctuationTest : public celeritas_test::MockTestBase
{
  protected:
    void SetUp() override
    {
        fluctuation_ = std::make_shared<FluctuationParams>(this->particle(),
                                                           this->material());
    }

    std::shared_ptr<const FluctuationParams> fluctuation_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(FluctuationTest, data)
{
    const auto& urban = fluctuation_->host_ref().urban;

    {
        // Celerogen: Z=1, I=19.2 eV
        const auto& params = urban[MaterialId{0}];
        EXPECT_SOFT_EQ(1, params.oscillator_strength[0]);
        EXPECT_SOFT_EQ(0, params.oscillator_strength[1]);
        EXPECT_SOFT_EQ(19.2e-6, params.binding_energy[0]);
        EXPECT_SOFT_EQ(1e-5, params.binding_energy[1]);
    }
    {
        // Energy loss fluctuation model parameters
        const auto& params = urban[MaterialId{2}];
        EXPECT_SOFT_EQ(0.80582524271844658, params.oscillator_strength[0]);
        EXPECT_SOFT_EQ(0.1941747572815534, params.oscillator_strength[1]);
        EXPECT_SOFT_EQ(9.4193231228829647e-5, params.binding_energy[0]);
        EXPECT_SOFT_EQ(1.0609e-3, params.binding_energy[1]);
    }
}

// TODO: add tests for helper, delta, gaussian, urban
