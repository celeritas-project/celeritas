//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MscStep.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/Units.hh"
#include "celeritas/em/msc/MscStepUpdater.hh"

#include "celeritas_test.hh"

constexpr auto cm = celeritas::units::centimeter;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class MscStepTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override
    {
        EXPECT_SOFT_EQ(1e-7 * cm, msc_params_.min_step());
    }

    // Note that parameters are from Geant4@11.0.3 data, stainless steel, 10.01
    // MeV electron
    UrbanMscParameters msc_params_;
};

TEST_F(MscStepTest, high_energy)
{
    // From 51.0231 MeV electron
    const real_type prange{4.5639217207134};
    const real_type lambda{2.0538835907703e1};
    MscStep step;
    step.true_path = 0.5;  // True physics step limit
    step.geom_path = 0.4;  // Maximum geometry step
    {
        // Perhaps this occurs only when there are bugs in the xs data?
        SCOPED_TRACE("negative alpha");
        step.alpha = -0.01;
        MscStepUpdater geo_to_true(msc_params_, step, range_, lambda_);

        EXPECT_SOFT_EQ(5e-8 * cm, geo_to_true(5e-8 * cm));
        EXPECT_SOFT_EQ(1e-8 * cm, geo_to_true(1e-8 * cm));
        EXPECT_SOFT_EQ(0.10682248885172108 * cm, geo_to_true(0.1 * cm));
        EXPECT_SOFT_EQ(step.true_path - 1e-10,
                       geo_to_true(step.geom_path - 1e-10));
        EXPECT_SOFT_EQ(step.true_path, geo_to_true(step.geom_path));
    }
    {
        // Perhaps this occurs only when there are bugs in the xs data?
        SCOPED_TRACE("zero (flag) alpha");
        step.alpha = MscStep::small_step_alpha();
        MscStepUpdater geo_to_true(msc_params_, step, range_, lambda_);

        EXPECT_SOFT_EQ(5e-8 * cm, geo_to_true(5e-8 * cm));
        EXPECT_SOFT_EQ(0.3 * cm, geo_to_true(0.3 * cm));
        EXPECT_SOFT_EQ(step.true_path - 1e-10,
                       geo_to_true(step.geom_path - 1e-10));
        EXPECT_SOFT_EQ(step.true_path, geo_to_true(step.geom_path));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
