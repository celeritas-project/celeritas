//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalRayleigh.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/model/RayleighInteractor.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;

//---------------------------------------------------------------------------//

class RayleighInteractorTest : public InteractorHostTestBase
{
  protected:
    void SetUp() override
    {
        inc_direction_ = {0, 0, 1};

        // Check incident quantities are valid
        sanity_check(inc_direction_, track_view().polarization());
    }

    void sanity_check(Real3 const& direction, Real3 const& polarization) const
    {
        // Check that direction and polarization are unit vectors
        EXPECT_SOFT_EQ(1, norm(direction));
        EXPECT_SOFT_EQ(1, norm(polarization));
        // Check that direction and polarization are perpendicular
        EXPECT_SOFT_EQ(0, dot_product(direction, polarization));
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Interactions should always be scattering
        EXPECT_EQ(Interaction::Action::scattered, interaction.action);
        sanity_check(interaction.direction, interaction.polarization);
    }

    TrackView const& track_view() const
    {
        // Energy doesn't matter; initial polarization should be perpendicular
        static TrackView view{units::MevEnergy{13e-6}, {1, 0, 0}};
        return view;
    }

    Real3 inc_direction_;
};

TEST_F(RayleighInteractorTest, basic)
{
    int const num_samples = 4;

    RayleighInteractor interact{track_view(), inc_direction_};

    auto& rng_engine = this->rng();

    std::vector<real_type> dir_angle;
    std::vector<real_type> pol_angle;

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        this->sanity_check(result);

        dir_angle.push_back(dot_product(result.direction, inc_direction_));
        pol_angle.push_back(
            dot_product(result.polarization, track_view().polarization()));
    }

    static double const expected_dir_angle[] = {-0.38366589898599,
                                                -0.18253767807648,
                                                0.42140775018143,
                                                -0.15366976713254};
    static double const expected_pol_angle[] = {
        0.46914077892327, 0.31484033374691, 0.82815845476385, 0.50516788333603};

    EXPECT_VEC_SOFT_EQ(expected_dir_angle, dir_angle);
    EXPECT_VEC_SOFT_EQ(expected_pol_angle, pol_angle);
}

TEST_F(RayleighInteractorTest, stress_test)
{
    int const num_samples = 1'000;

    RayleighInteractor interact{track_view(), inc_direction_};

    auto& rng_engine = this->rng();

    real_type dir_angle_1st_moment = 0;
    real_type dir_angle_2nd_moment = 0;

    real_type pol_angle_1st_moment = 0;
    real_type pol_angle_2nd_moment = 0;

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        this->sanity_check(result);

        real_type dir_angle = dot_product(result.direction, inc_direction_);
        dir_angle_1st_moment += dir_angle;
        dir_angle_2nd_moment += dir_angle * dir_angle;

        real_type pol_angle
            = dot_product(result.polarization, track_view().polarization());
        pol_angle_1st_moment += pol_angle;
        pol_angle_2nd_moment += pol_angle * pol_angle;
    }

    dir_angle_1st_moment /= num_samples;
    dir_angle_2nd_moment /= num_samples;
    pol_angle_1st_moment /= num_samples;
    pol_angle_2nd_moment /= num_samples;

    static double const expected_dir_angle_1st_moment = -0.0037666175399340422;
    static double const expected_dir_angle_2nd_moment = 0.19865973173404519;
    static double const expected_pol_angle_1st_moment = -0.020707206147726396;
    static double const expected_pol_angle_2nd_moment = 0.40045856568225996;

    EXPECT_SOFT_EQ(expected_dir_angle_1st_moment, dir_angle_1st_moment);
    EXPECT_SOFT_EQ(expected_dir_angle_2nd_moment, dir_angle_2nd_moment);
    EXPECT_SOFT_EQ(expected_pol_angle_1st_moment, pol_angle_1st_moment);
    EXPECT_SOFT_EQ(expected_pol_angle_2nd_moment, pol_angle_2nd_moment);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
