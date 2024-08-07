//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalRayleigh.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/interactor/RayleighInteractor.hh"

#include "InteractorHostTestBase.hh"
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
        // Check incident quantities are valid
        this->check_direction_polarization(
            this->direction(), this->photon_track().polarization());
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Interactions should always be scattering
        EXPECT_EQ(Interaction::Action::scattered, interaction.action);
        this->check_direction_polarization(interaction);
    }
};

TEST_F(RayleighInteractorTest, basic)
{
    int const num_samples = 4;

    RayleighInteractor interact{this->photon_track(), this->direction()};

    auto& rng_engine = this->rng();

    std::vector<real_type> dir_angle;
    std::vector<real_type> pol_angle;

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        this->sanity_check(result);

        dir_angle.push_back(dot_product(result.direction, this->direction()));
        pol_angle.push_back(dot_product(result.polarization,
                                        this->photon_track().polarization()));
    }

    static real_type const expected_dir_angle[] = {
        -0.38366589898599,
        -0.18253767807648,
        0.42140775018143,
        -0.15366976713254,
    };
    static real_type const expected_pol_angle[] = {
        0.46914077892327,
        0.31484033374691,
        0.82815845476385,
        0.50516788333603,
    };

    EXPECT_EQ(136, rng_engine.count());
    EXPECT_VEC_SOFT_EQ(expected_dir_angle, dir_angle);
    EXPECT_VEC_SOFT_EQ(expected_pol_angle, pol_angle);
}

TEST_F(RayleighInteractorTest, stress_test)
{
    int const num_samples = 1'000;

    RayleighInteractor interact{this->photon_track(), this->direction()};

    auto& rng_engine = this->rng();

    Array<real_type, 2> dir_moment = {0, 0};
    Array<real_type, 2> pol_moment = {0, 0};

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        this->sanity_check(result);

        real_type dir_angle = dot_product(result.direction, this->direction());
        dir_moment[0] += dir_angle;
        dir_moment[1] += ipow<2>(dir_angle);

        real_type pol_angle = dot_product(result.polarization,
                                          this->photon_track().polarization());
        pol_moment[0] += pol_angle;
        pol_moment[1] += ipow<2>(pol_angle);
    }

    dir_moment /= num_samples;
    pol_moment /= num_samples;

    PRINT_EXPECTED(dir_moment);
    PRINT_EXPECTED(pol_moment);

    static real_type const expected_dir_moment[]
        = {-0.0037666175399340422, 0.19865973173404519};
    static real_type const expected_pol_moment[]
        = {-0.020707206147726396, 0.40045856568225996};

    EXPECT_VEC_SOFT_EQ(expected_dir_moment, dir_moment);
    EXPECT_VEC_SOFT_EQ(expected_pol_moment, pol_moment);
    EXPECT_EQ(24400, rng_engine.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
