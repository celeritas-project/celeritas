//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalRayleigh.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/model/OpticalRayleighInteractor.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class OpticalRayleighInteractorTest : public InteractorHostTestBase
{
  protected:
    void SetUp() override
    {
        inc_direction_ = {0, 0, 1};

        sanity_check(inc_direction_, track_view().polarization());
    }

    void sanity_check(Real3 const& direction, Real3 const& polarization) const
    {
        EXPECT_SOFT_EQ(1, norm(direction));
        EXPECT_SOFT_EQ(1, norm(polarization));
        EXPECT_SOFT_EQ(0, dot_product(direction, polarization));
    }

    void sanity_check(OpticalInteraction const& interaction) const
    {
        EXPECT_EQ(OpticalInteraction::Action::scattered, interaction.action);
        sanity_check(interaction.direction, interaction.polarization);
    }

    OpticalTrackView const& track_view() const
    {
        static OpticalTrackView view{units::MevEnergy{13e-6}, {1, 0, 0}};
        return view;
    }

    Real3 inc_direction_;
};

TEST_F(OpticalRayleighInteractorTest, basic)
{
    int const num_samples = 4;

    OpticalRayleighInteractor interact{track_view(), inc_direction_};

    auto& rng_engine = this->rng();

    std::vector<real_type> dir_angle;
    std::vector<real_type> pol_angle;

    for (int i : range(num_samples))
    {
        OpticalInteraction result = interact(rng_engine);
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

TEST_F(OpticalRayleighInteractorTest, stress_test)
{
    int const num_samples = 1'000;

    OpticalRayleighInteractor interact{track_view(), inc_direction_};

    auto& rng_engine = this->rng();

    real_type dir_angle_1st_moment = 0;
    real_type dir_angle_2nd_moment = 0;

    real_type pol_angle_1st_moment = 0;
    real_type pol_angle_2nd_moment = 0;

    for (int i : range(num_samples))
    {
        OpticalInteraction result = interact(rng_engine);
        this->sanity_check(result);

        real_type dir_angle = dot_product(result.direction, inc_direction_);
        dir_angle_1st_moment += dir_angle;
        dir_angle_2nd_moment += dir_angle * dir_angle;

        real_type pol_angle
            = dot_product(result.polarization, track_view().polarization());
        pol_angle_1st_moment += pol_angle;
        pol_angle_2nd_moment += pol_angle * pol_angle;
    }

    static double const expected_dir_angle_1st_moment = -3.766617539934042;
    static double const expected_dir_angle_2nd_moment = 198.65973173404518;
    static double const expected_pol_angle_1st_moment = -20.707206147726396;
    static double const expected_pol_angle_2nd_moment = 400.45856568225997;

    EXPECT_SOFT_EQ(expected_dir_angle_1st_moment, dir_angle_1st_moment);
    EXPECT_SOFT_EQ(expected_dir_angle_2nd_moment, dir_angle_2nd_moment);
    EXPECT_SOFT_EQ(expected_pol_angle_1st_moment, pol_angle_1st_moment);
    EXPECT_SOFT_EQ(expected_pol_angle_2nd_moment, pol_angle_2nd_moment);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
