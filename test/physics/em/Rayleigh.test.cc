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
        const auto& particles = this->particle_params();
        pointers_.gamma_id    = particles.find(pdg::gamma());

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

        // Construct RayleighModel and set the host data pointers
        model_ = std::make_shared<RayleighModel>(
            ModelId{0}, particles, this->material_params());
        pointers_ = model_->host_pointers();
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);

        // Check change to parent track
        EXPECT_EQ(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_EQ(1.0, interaction.energy.value());
        EXPECT_EQ(celeritas::Action::scattered, interaction.action);
    }

  protected:
    std::shared_ptr<RayleighModel>            model_;
    celeritas::detail::RayleighNativePointers pointers_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RayleighInteractorTest, basic)
{
    const int num_samples = 4;

    // Sample an element
    ElementId el_id{0};

    // Create the interactor
    RayleighInteractor interact(this->model_->host_pointers(),
                                this->particle_track(),
                                this->direction(),
                                el_id);

    RandomEngine& rng_engine = this->rng();

    // Produce four samples from the original/incident photon
    std::vector<double> angle;

    for (CELER_MAYBE_UNUSED auto i : celeritas::range(num_samples))
    {
        celeritas::Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        angle.push_back(dot_product(result.direction, this->direction()));
    }

    const double expected_angle[] = {-0.0798080588007317,
                                     -0.199218754040088,
                                     -0.420659925857054,
                                     0.464669752940999};

    EXPECT_VEC_SOFT_EQ(expected_angle, angle);
}

TEST_F(RayleighInteractorTest, stress_test)
{
    const int num_samples = 8192;

    // Sample an element
    ElementId el_id{0};

    // Create the interactor
    RayleighInteractor interact(this->model_->host_pointers(),
                                this->particle_track(),
                                this->direction(),
                                el_id);

    RandomEngine& rng_engine = this->rng();

    // Produce four samples from the original/incident photon
    for (CELER_MAYBE_UNUSED auto i : celeritas::range(num_samples))
    {
        celeritas::Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);
    }
}
