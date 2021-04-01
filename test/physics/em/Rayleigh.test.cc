//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighInteractor.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/detail/RayleighInteractor.hh"
#include "physics/em/detail/Rayleigh.hh"
#include "physics/em/RayleighModel.hh"

#include "physics/material/ElementView.hh"
#include "physics/material/Types.hh"
#include "physics/base/Units.hh"
#include "physics/material/MaterialTrackView.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "physics/base/Units.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::ElementId;
using celeritas::MaterialParams;
using celeritas::RayleighModel;
using celeritas::detail::RayleighInteractor;
using celeritas::units::AmuMass;

namespace constants = celeritas::constants;
namespace pdg       = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RayleighInteractorTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using celeritas::ParticleDef;
        using namespace celeritas::units;
        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleDef::stable_decay_constant();

        // Set up shared particle data for RayleighModel
        Base::set_particle_params(
            {{"gamma", pdg::gamma(), zero, zero, stable}});
        const auto& particles = *this->particle_params();
        group_.gamma_id       = particles.find(pdg::gamma());

        // Set default particle to incident 1 MeV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{1.0});
        this->set_inc_direction({0, 0, 1});

        // Setup MaterialView
        MaterialParams::Input inp;
        inp.elements  = {{8, AmuMass{15.999}, "O"},
                        {74, AmuMass{183.84}, "W"},
                        {82, AmuMass{207.2}, "Pb"}};
        inp.materials = {
            {1.0 * constants::na_avogadro,
             293.0,
             celeritas::MatterState::solid,
             {{celeritas::ElementId{0}, 0.5},
              {celeritas::ElementId{1}, 0.3},
              {celeritas::ElementId{2}, 0.2}},
             "PbWO"},
        };
        this->set_material_params(inp);
        this->set_material("PbWO");

        // Construct RayleighModel and set the host data group
        model_ = std::make_shared<RayleighModel>(
            ModelId{0}, particles, *this->material_params());
        group_ = model_->host_group();
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);

        // Check change to parent track - coherent scattering
        EXPECT_EQ(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_EQ(celeritas::Action::scattered, interaction.action);
    }

  protected:
    std::shared_ptr<RayleighModel>       model_;
    celeritas::detail::RayleighNativeRef group_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RayleighInteractorTest, basic)
{
    // Sample an element (TODO: add ElementSelector)
    ElementId el_id{0};

    std::vector<real_type>         angle;
    std::vector<unsigned long int> rng_counts;

    // Sample scattering angle and count rng used for each incident energy
    RandomEngine& rng_engine = this->rng();

    for (double inc_e : {0.05, 0.1, 1.0, 10.0, 100.0})
    {
        // Set the incident particle energy
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});

        // Reset rng count
        rng_engine.reset_count();

        // Create the interactor
        RayleighInteractor interact(this->model_->host_group(),
                                    this->particle_track(),
                                    this->direction(),
                                    el_id);


        // Produce a sample from the original/incident photon
        celeritas::Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);
        rng_engine.count();

        angle.push_back(dot_product(result.direction, this->direction()));
        rng_counts.push_back(rng_engine.count());
    }

    const real_type expected_angle[] = {-0.726848858395344,
                                        -0.95887836792401,
                                        -0.910276761243175,
                                        -0.223090692143936,
                                        -0.388188923742863};

    const unsigned long int expected_rng_counts[] = {17774, 1568, 74, 8, 14};

    EXPECT_VEC_SOFT_EQ(expected_angle, angle);
    EXPECT_VEC_EQ(expected_rng_counts, rng_counts);
}

TEST_F(RayleighInteractorTest, stress_test)
{
    const int num_samples = 1028;

    // Sample an element
    ElementId el_id{0};

    std::vector<real_type>         average_angle;
    std::vector<real_type>         average_rng_counts;

    // Sample scattering angle and count rng used for each incident energy
    RandomEngine& rng_engine = this->rng();

    for (double inc_e : {0.05, 0.1, 1.0, 10.0, 100.0})
    {
        // Set the incident particle energy
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});

        // Reset the rng counter
        rng_engine.reset_count();

        // Create the interactor
        RayleighInteractor interact(this->model_->host_group(),
                                    this->particle_track(),
                                    this->direction(),
                                    el_id);

        // Produce num_samples from the original/incident photon
        real_type sum_angle = 0;
        for (CELER_MAYBE_UNUSED auto i : celeritas::range(num_samples))
        {
            celeritas::Interaction result = interact(rng_engine);
            SCOPED_TRACE(result);
            this->sanity_check(result);
            rng_engine.count();
            sum_angle += dot_product(result.direction, this->direction());
        }

        average_rng_counts.push_back(rng_engine.count() / num_samples);
        average_angle.push_back(sum_angle / num_samples);
    }

    const real_type expected_average_rng_counts[] = {49710, 11914, 137, 12, 10};

    const real_type expected_average_angle[] = {-0.003575422006847,
                                                0.03399904936658,
                                                -0.01054842299125,
                                                0.008589239285608,
                                                -0.01178001515854};

    EXPECT_VEC_SOFT_EQ(expected_average_rng_counts, average_rng_counts);
    EXPECT_VEC_SOFT_EQ(expected_average_angle, average_angle);
}
