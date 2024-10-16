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
        data_.ids.electron_neutrino = params.find(pdg::electron_neutrino());
        data_.ids.anti_electron_neutrino
            = params.find(pdg::anti_electron_neutrino());
        data_.ids.muon_neutrino = params.find(pdg::mu_neutrino());
        data_.ids.anti_muon_neutrino = params.find(pdg::anti_mu_neutrino());
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
    this->set_inc_direction({0, 0, 1});
    auto const at_rest = MevEnergy{0};

    // Anti-muon decay
    {
        this->set_inc_particle(pdg::mu_plus(), at_rest);

        MuDecayInteractor interact(data_,
                                   this->particle_track(),
                                   this->direction(),
                                   this->secondary_allocator());
        auto result = interact(this->rng());

        EXPECT_EQ(Interaction::Action::decay, result.action);
        EXPECT_EQ(pdg::positron(),
                  params.id_to_pdg(result.secondaries[0].particle_id));
        EXPECT_EQ(pdg::electron_neutrino(),
                  params.id_to_pdg(result.secondaries[1].particle_id));
        EXPECT_EQ(pdg::anti_mu_neutrino(),
                  params.id_to_pdg(result.secondaries[2].particle_id));
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
        EXPECT_EQ(3, sec.size());
        EXPECT_EQ(pdg::electron(), params.id_to_pdg(sec[0].particle_id));
        EXPECT_EQ(pdg::anti_electron_neutrino(),
                  params.id_to_pdg(sec[1].particle_id));
        EXPECT_EQ(pdg::mu_neutrino(), params.id_to_pdg(sec[2].particle_id));

        // Check energy and momentum conservation
        double secondary_cm_energy{};
        for (auto i : range(3))
        {
            secondary_cm_energy += sec[i].energy.value();
        }
        EXPECT_SOFT_EQ(data_.muon_mass, secondary_cm_energy);

        Real3 total_momentum{};
        for (auto dir : range(3))
        {
            for (auto const& part : sec)
            {
                total_momentum[dir] += part.direction[dir]
                                       * part.energy.value();
            }
        }
        EXPECT_VEC_NEAR(Real3({0, 0, 0}), total_momentum, 1e-11);
    }
}

//---------------------------------------------------------------------------//
TEST_F(MuDecayInteractorTest, stress_test)
{
    size_type const num_samples = 10000;
    MevEnergy one_gev{1000};
    this->resize_secondaries(3 * num_samples);
    this->set_inc_particle(pdg::mu_minus(), one_gev);
    this->set_inc_direction({0, 0, 1});

    MuDecayInteractor interact(data_,
                               this->particle_track(),
                               this->direction(),
                               this->secondary_allocator());

    real_type avg_sec_energies[3]{};  // Average energy per secondary
    Real3 avg_total_momentum{};  // Average total momentum per decay

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

    for (auto j : range(3))
    {
        avg_sec_energies[j] /= num_samples;
        avg_total_momentum[j] /= num_samples;
    }

    // Average energies should add up to ~1 GeV
    static double const expected_avg_sec_energies[]
        = {358.09364458255, 301.62416070388, 350.20342374459};

    // Average total momentum for all secondaries in each direction should be
    // close to the initial muon momentum
    static double const expected_avg_total_momentum[]
        = {-0.0013733608217145, -0.0025640101479541, 1004.3436429006};

    EXPECT_VEC_SOFT_EQ(expected_avg_sec_energies, avg_sec_energies);
    EXPECT_VEC_SOFT_EQ(expected_avg_total_momentum, avg_total_momentum);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
