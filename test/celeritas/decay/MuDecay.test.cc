//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/MuDecay.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/decay/interactor/MuDecayInteractor.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class MuDecayInteractorTest : public InteractorHostTestBase
{
  protected:
    void SetUp() override
    {
        auto const& params = *this->particle_params();
        data_.ids.electron = params.find(pdg::electron());
        data_.ids.positron = params.find(pdg::positron());
        data_.ids.mu_minus = params.find(pdg::mu_minus());
        data_.ids.mu_plus = params.find(pdg::mu_plus());
        data_.electron_mass = params.get(data_.ids.electron).mass().value();
        data_.muon_mass = params.get(data_.ids.mu_minus).mass().value();
    }

  protected:
    MuDecayData data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MuDecayInteractorTest, basic)
{
    auto const& params = *this->particle_params();
    auto const& muon = params.get(data_.ids.mu_minus);
    EXPECT_GE(1, muon.decay_constant());  // FIXME

    this->set_inc_direction({0, 0, 1});
    auto const hundred_mev = MevEnergy{100};

    // Anti-muon
    {
        this->set_inc_particle(pdg::mu_plus(), hundred_mev);

        MuDecayInteractor interact(data_,
                                   this->particle_track(),
                                   this->direction(),
                                   this->secondary_allocator());
        auto result = interact(this->rng());
        EXPECT_EQ(Interaction::Action::decay, result.action);
        EXPECT_EQ(pdg::positron(),
                  params.id_to_pdg(result.secondaries[0].particle_id));
    }

    // Muon
    {
        this->set_inc_particle(pdg::mu_minus(), hundred_mev);

        MuDecayInteractor interact(data_,
                                   this->particle_track(),
                                   this->direction(),
                                   this->secondary_allocator());
        auto result = interact(this->rng());
        EXPECT_EQ(Interaction::Action::decay, result.action);

        auto const& sec = result.secondaries;
        EXPECT_EQ(3, sec.size());
        EXPECT_EQ(pdg::electron(), params.id_to_pdg(sec[0].particle_id));
    }
}

//---------------------------------------------------------------------------//
TEST_F(MuDecayInteractorTest, stress_test)
{
    size_type const num_samples = 1000;
    this->resize_secondaries(3 * num_samples);
    this->set_inc_particle(pdg::mu_minus(), MevEnergy{1});
    this->set_inc_direction({0, 0, 1});

    MuDecayInteractor interact(data_,
                               this->particle_track(),
                               this->direction(),
                               this->secondary_allocator());
    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto r = interact(this->rng());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
