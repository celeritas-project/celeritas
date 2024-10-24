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
        data_.electron_mass = params.get(data_.ids.electron).mass();
        data_.muon_mass = params.get(data_.ids.mu_minus).mass();

        this->set_inc_direction({0, 0, 1});
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
    auto const at_rest = MevEnergy{0};
    auto const max_lepton_energy = real_type{0.5} * data_.muon_mass.value()
                                   - data_.electron_mass.value();

    // Anti-muon decay
    {
        this->set_inc_particle(pdg::mu_plus(), at_rest);
        MuDecayInteractor interact(data_,
                                   this->particle_track(),
                                   this->direction(),
                                   this->secondary_allocator());
        auto result = interact(this->rng());

        EXPECT_EQ(Interaction::Action::decay, result.action);
        auto const& sec = result.secondaries;
        EXPECT_EQ(1, sec.size());
        EXPECT_EQ(pdg::positron(), params.id_to_pdg(sec[0].particle_id));
        EXPECT_GE(max_lepton_energy, sec[0].energy.value());
    }

    // Muon decay
    {
        this->set_inc_particle(pdg::mu_minus(), at_rest);
        MuDecayInteractor interact(data_,
                                   this->particle_track(),
                                   this->direction(),
                                   this->secondary_allocator());
        auto result = interact(this->rng());

        EXPECT_EQ(Interaction::Action::decay, result.action);
        auto const& sec = result.secondaries;
        EXPECT_EQ(1, sec.size());
        EXPECT_EQ(pdg::electron(), params.id_to_pdg(sec[0].particle_id));
        EXPECT_GE(max_lepton_energy, sec[0].energy.value());
    }
}

//---------------------------------------------------------------------------//
TEST_F(MuDecayInteractorTest, stress_test)
{
    size_type const num_secondaries = 1;
    size_type const num_samples = 10000;
    MevEnergy const one_gev{1000};

    this->resize_secondaries(num_secondaries * num_samples);
    this->set_inc_particle(pdg::mu_minus(), one_gev);
    MuDecayInteractor interact(data_,
                               this->particle_track(),
                               this->direction(),
                               this->secondary_allocator());

    double avg_sec_energies[num_secondaries]{};  // Avg energy per secondary
    Real3 avg_total_momentum{};  // Avg total momentum per decay

    for ([[maybe_unused]] auto sample : range(num_samples))
    {
        auto result = interact(this->rng());
        auto const& sec = result.secondaries;

        for (auto i : range(sec.size()))
        {
            auto const& part = sec[i];
            avg_sec_energies[i] += part.energy.value();
            for (auto j : range(3))
            {
                avg_total_momentum[j] += part.direction[j]
                                         * part.energy.value();
            }
        }
    }

    for (auto j : range(num_secondaries))
    {
        avg_sec_energies[j] /= num_samples;
    }

    for (auto j : range(3))
    {
        avg_total_momentum[j] /= num_samples;
    }

    // With only one secondary being returned, there is no expectation of
    // energy or momentum conservation
    static double const expected_avg_sec_energies[] = {384.1448835314};
    static double const expected_avg_total_momentum[]
        = {0.554857155437642, -0.113397931984889, 382.358304534532};

    EXPECT_VEC_SOFT_EQ(expected_avg_sec_energies, avg_sec_energies);
    EXPECT_VEC_SOFT_EQ(expected_avg_total_momentum, avg_total_momentum);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
