//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/DDMucfInteractor.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/mucf/interactor/DDMucfInteractor.hh"

#include "corecel/cont/Range.hh"

#include "MucfInteractorHostTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class DDMucfInteractorTest : public MucfInteractorHostTestBase
{
    using Base = MucfInteractorHostTestBase;
    using Channel = DDMucfInteractor::Channel;
    using MevEnergy = units::MevEnergy;
    using MevMass = units::MevMass;

  protected:
    void SetUp() override
    {
        // At-rest muon primary
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{0.0});
        this->set_inc_direction({1, 0, 0});
        data_ = this->host_data();
    }

    // Detailed validation of the interaction result
    void
    validate_interaction(Interaction const& interaction, Channel channel) const
    {
        EXPECT_LT(channel, Channel::size_);

        // Primary muon should be killed
        EXPECT_EQ(Action::absorbed, interaction.action);

        auto const& sec = interaction.secondaries;

        // Verify channel-specific data
        if (channel == Channel::helium3_muon_neutron)
        {
            ASSERT_EQ(num_secondaries_[channel], sec.size());

            // First particle is the outgoing neutron with 0.75 * 3.3 MeV
            EXPECT_EQ(data_.particle_ids.neutron, sec[0].particle_id);
            EXPECT_SOFT_EQ(0.75 * 3.3, sec[0].energy.value());

            // Check particles
            EXPECT_EQ(data_.particle_ids.mu_minus, sec[1].particle_id);
            EXPECT_EQ(data_.particle_ids.he3, sec[2].particle_id);

            // Check approximate energy conservation
            real_type total_kinetic_energy = 0;
            for (auto const& s : interaction.secondaries)
            {
                total_kinetic_energy += s.energy.value();
            }
            // The total kinetic energy range is within [3.2, 3.5] MeV
            EXPECT_SOFT_NEAR(3.3, total_kinetic_energy, 0.3);

            // Check momentum conservation
            auto const neutron_p_mag = this->calc_momentum(
                sec[0].energy, data_.particle_masses.neutron);
            auto const muon_p_mag = this->calc_momentum(
                sec[1].energy, data_.particle_masses.mu_minus);

            Real3 he3_momentum, total_momentum;
            for (auto i : range(3))
            {
                real_type neutron_momentum_i = sec[0].direction[i]
                                               * neutron_p_mag;
                real_type muon_momentum_i = sec[1].direction[i] * muon_p_mag;
                he3_momentum[i] = -(neutron_momentum_i + muon_momentum_i);
                total_momentum[i] = neutron_momentum_i + muon_momentum_i
                                    + he3_momentum[i];
            }

            EXPECT_VEC_SOFT_EQ(sec[2].direction,
                               make_unit_vector(he3_momentum));
            EXPECT_VEC_SOFT_EQ(Real3{}, total_momentum);
        }

        else if (channel == Channel::muonichelium3_neutron)
        {
            ASSERT_EQ(num_secondaries_[channel],
                      interaction.secondaries.size());

            // Check particle types
            EXPECT_EQ(data_.particle_ids.neutron,
                      interaction.secondaries[0].particle_id);
            EXPECT_EQ(data_.particle_ids.muonic_he3,
                      interaction.secondaries[1].particle_id);

            // First particle is the outgoing neutron with 0.75 * 3.3 MeV
            EXPECT_SOFT_EQ(0.75 * 3.3,
                           interaction.secondaries[0].energy.value());

            // Check directions are opposite
            EXPECT_SOFT_EQ(-1.0,
                           dot_product(interaction.secondaries[0].direction,
                                       interaction.secondaries[1].direction));
        }

        else if (channel == Channel::tritium_muon_proton)
        {
            ASSERT_EQ(num_secondaries_[channel], sec.size());

            // First particle is the outgoing proton with 0.75 * 4.03 MeV
            EXPECT_EQ(data_.particle_ids.proton, sec[0].particle_id);
            EXPECT_SOFT_EQ(0.75 * 4.03, sec[0].energy.value());

            // Check particles
            EXPECT_EQ(data_.particle_ids.mu_minus, sec[1].particle_id);
            EXPECT_EQ(data_.particle_ids.triton, sec[2].particle_id);

            // Check approximate energy conservation
            // The total kinetic energy is only very roughly 4.03 MeV due
            // to simplistic sampling.
            real_type total_kinetic_energy = 0;
            for (auto const& s : interaction.secondaries)
            {
                total_kinetic_energy += s.energy.value();
            }
            EXPECT_SOFT_NEAR(4.03, total_kinetic_energy, 0.5);

            // Check momentum conservation
            auto const proton_p_mag = this->calc_momentum(
                sec[0].energy, data_.particle_masses.proton);
            auto const muon_p_mag = this->calc_momentum(
                sec[1].energy, data_.particle_masses.mu_minus);

            Real3 triton_momentum, total_momentum;
            for (auto i : range(3))
            {
                real_type proton_momentum_i = sec[0].direction[i]
                                              * proton_p_mag;
                real_type muon_momentum_i = sec[1].direction[i] * muon_p_mag;
                triton_momentum[i] = -(proton_momentum_i + muon_momentum_i);
                total_momentum[i] = proton_momentum_i + muon_momentum_i
                                    + triton_momentum[i];
            }

            EXPECT_VEC_SOFT_EQ(sec[2].direction,
                               make_unit_vector(triton_momentum));
            EXPECT_VEC_SOFT_EQ(Real3{}, total_momentum);
        }
    }

    // Momentum magnitude (p = \sqrt{K^2 + 2mK})
    real_type calc_momentum(MevEnergy energy, MevMass mass) const
    {
        return std::sqrt(ipow<2>(value_as<MevEnergy>(energy))
                         + 2 * value_as<MevMass>(mass)
                               * value_as<MevEnergy>(energy));
    }

  protected:
    HostCRef<DTMixMucfData> data_;
    EnumArray<Channel, size_type> num_secondaries_{
        3,  // helium3_muon_neutron
        2,  // muonichelium3_neutron
        3  // tritium_muon_proton
    };
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(DDMucfInteractorTest, helium3_muon_neutron)
{
    auto const channel = DDMucfInteractor::Channel::helium3_muon_neutron;

    size_type const num_samples = 4;
    this->resize_secondaries(num_samples * num_secondaries_[channel]);
    auto& rng = this->rng();

    DDMucfInteractor interact(data_, channel, this->secondary_allocator());
    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto result = interact(rng);
        this->validate_interaction(result, channel);
    }
}

//---------------------------------------------------------------------------//
TEST_F(DDMucfInteractorTest, muonichelium3_neutron)
{
    auto const channel = DDMucfInteractor::Channel::muonichelium3_neutron;

    size_type const num_samples = 4;
    this->resize_secondaries(num_samples * num_secondaries_[channel]);
    auto& rng = this->rng();

    DDMucfInteractor interact(data_, channel, this->secondary_allocator());
    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto result = interact(rng);
        this->validate_interaction(result, channel);
    }
}

//---------------------------------------------------------------------------//
TEST_F(DDMucfInteractorTest, tritium_muon_proton)
{
    auto const channel = DDMucfInteractor::Channel::tritium_muon_proton;

    size_type const num_samples = 4;
    this->resize_secondaries(num_samples * num_secondaries_[channel]);
    auto& rng = this->rng();

    DDMucfInteractor interact(data_, channel, this->secondary_allocator());
    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto result = interact(rng);
        this->validate_interaction(result, channel);
    }
}

//---------------------------------------------------------------------------//
TEST_F(DDMucfInteractorTest, stress_test)
{
    size_type const num_samples = 10000;
    real_type total_avg_secondaries{0};

    for (auto channel : {DDMucfInteractor::Channel::helium3_muon_neutron,
                         DDMucfInteractor::Channel::muonichelium3_neutron,
                         DDMucfInteractor::Channel::tritium_muon_proton})
    {
        this->resize_secondaries(num_samples * num_secondaries_[channel]);
        DDMucfInteractor interact(data_, channel, this->secondary_allocator());

        auto& rng = this->rng();
        for ([[maybe_unused]] auto i : range(num_samples))
        {
            auto result = interact(rng);
            total_avg_secondaries += result.secondaries.size();
        }
    }
    total_avg_secondaries /= 3 * num_samples;  // Average over all channels

    // (3 + 2 + 3) / 3
    static real_type const expected_total_avg_secondaries{8. / 3.};
    EXPECT_SOFT_EQ(expected_total_avg_secondaries, total_avg_secondaries);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
