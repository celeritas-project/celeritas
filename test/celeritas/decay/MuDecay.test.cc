//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/MuDecay.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
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
        data_.electron_id = params.find(pdg::electron());
        data_.positron_id = params.find(pdg::positron());
        data_.mu_minus_id = params.find(pdg::mu_minus());
        data_.mu_plus_id = params.find(pdg::mu_plus());
        data_.electron_mass = params.get(data_.electron_id).mass();
        data_.muon_mass = params.get(data_.mu_minus_id).mass();

        this->set_inc_direction({0, 0, 1});
    }

  protected:
    MuDecayData data_;

    struct TestResult
    {
        double avg_sec_energy{};  // Avg energy of the outgoing electron
        Real3 avg_total_momentum{};  // Avg total momentum per decay
    };

    TestResult loop(size_type num_samples, MevEnergy energy)
    {
        this->resize_secondaries(num_samples);
        this->set_inc_particle(pdg::mu_minus(), energy);
        MuDecayInteractor interact(data_,
                                   this->particle_track(),
                                   this->direction(),
                                   this->secondary_allocator());

        TestResult tr;
        for ([[maybe_unused]] auto sample : range(num_samples))
        {
            auto result = interact(this->rng());
            auto const& part = result.secondaries[0];

            tr.avg_sec_energy += part.energy.value();
            axpy(part.energy.value(), part.direction, &tr.avg_total_momentum);
        }
        tr.avg_sec_energy /= num_samples;
        tr.avg_total_momentum /= num_samples;

        return tr;
    }
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

        EXPECT_EQ(Interaction::Action::absorbed, result.action);
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

        EXPECT_EQ(Interaction::Action::absorbed, result.action);
        auto const& sec = result.secondaries;
        EXPECT_EQ(1, sec.size());
        EXPECT_EQ(pdg::electron(), params.id_to_pdg(sec[0].particle_id));
        EXPECT_GE(max_lepton_energy, sec[0].energy.value());
    }
}

//---------------------------------------------------------------------------//
TEST_F(MuDecayInteractorTest, stress_test)
{
    // With only one secondary being returned, there is no expectation of
    // energy or momentum conservation

    size_type const num_samples = 10000;

    // Muon with 1 GeV
    auto res_1gev = this->loop(num_samples, MevEnergy{1000});

    static double const expected_one_gev_avg_sec_energy = 384.834176348064;
    static double const expected_one_gev_avg_total_momentum[]
        = {0.36972198353269, 0.10949440149464, 383.06404348763};

    EXPECT_REAL_EQ(expected_one_gev_avg_sec_energy, res_1gev.avg_sec_energy);
    EXPECT_VEC_SOFT_EQ(expected_one_gev_avg_total_momentum,
                       res_1gev.avg_total_momentum);

    // Muon with 1 MeV
    auto res_1mev = this->loop(num_samples, MevEnergy{1});

    static double const expected_one_mev_avg_sec_energy = 36.891223054430647;
    static double const expected_one_mev_avg_total_momentum[]
        = {0.31976610917125, -0.13376923570472, 5.5275940366016};

    EXPECT_REAL_EQ(expected_one_mev_avg_sec_energy, res_1mev.avg_sec_energy);
    EXPECT_VEC_SOFT_EQ(expected_one_mev_avg_total_momentum,
                       res_1mev.avg_total_momentum);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
