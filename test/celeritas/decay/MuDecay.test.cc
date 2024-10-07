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

    // Muon params data
    {
        auto const& muon = params.get(data_.ids.mu_minus);
        EXPECT_SOFT_EQ(105.6583745, muon.mass().value());
        EXPECT_SOFT_EQ(1 / 2.1969811e-6, muon.decay_constant());
    }

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
    this->set_inc_particle(pdg::mu_minus(), MevEnergy{1000});
    this->set_inc_direction({0, 0, 1});

    MuDecayInteractor interact(data_,
                               this->particle_track(),
                               this->direction(),
                               this->secondary_allocator());

    real_type avg_tot_energy{0};  // Average energy per decay
    real_type avg_sec_energies[3] = {};  // Average energy per secondary
    real_type avg_sec_dirz[3] = {};  // Average z direction per secondary

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto result = interact(this->rng());
        auto const& sec = result.secondaries;

        for (auto j : range(3))
        {
            avg_sec_energies[j] += sec[j].energy.value();
            avg_tot_energy += sec[j].energy.value();
            avg_sec_dirz[j] += sec[j].direction[2];
        }
    }

    avg_tot_energy /= num_samples;
    for (auto j : range(3))
    {
        avg_sec_energies[j] /= num_samples;
        avg_sec_dirz[j] /= num_samples;
    }

    static double const expected_avg_sec_energies[]
        = {346.69889751678, 178.24258457901, 475.73040550737};
    static double const expected_avg_sec_dirz[]
        = {0.90003827924887, 0.81224792038635, 0.99099028265594};

    EXPECT_SOFT_EQ(1000.6718876031533, avg_tot_energy);
    EXPECT_VEC_SOFT_EQ(expected_avg_sec_energies, avg_sec_energies);
    EXPECT_VEC_SOFT_EQ(expected_avg_sec_dirz, avg_sec_dirz);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
