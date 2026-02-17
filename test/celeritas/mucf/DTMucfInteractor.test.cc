//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/DTMucfInteractor.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/mucf/interactor/DTMucfInteractor.hh"

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

class DTMucfInteractorTest : public MucfInteractorHostTestBase
{
    using Base = MucfInteractorHostTestBase;
    using Channel = DTMucfInteractor::Channel;
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

        // First particle is always an outgoing neutron with 14.1 MeV
        EXPECT_EQ(data_.particle_ids.neutron, sec[0].particle_id);
        EXPECT_SOFT_EQ(14.1, sec[0].energy.value());

        // Verify channel-specific data
        if (channel == Channel::alpha_muon_neutron)
        {
            ASSERT_EQ(num_secondaries_[channel], sec.size());

            // Check particles
            EXPECT_EQ(data_.particle_ids.mu_minus, sec[1].particle_id);
            EXPECT_EQ(data_.particle_ids.alpha, sec[2].particle_id);

            // Check approximate energy conservation
            // The total kinetic energy is only very roughly 17.6 MeV due to
            // simplistic sampling. See DTMucfInteractor documentation for
            // details.
            real_type total_kinetic_energy = 0;
            for (auto const& sec : interaction.secondaries)
            {
                total_kinetic_energy += sec.energy.value();
            }
            EXPECT_SOFT_NEAR(17.6, total_kinetic_energy, 0.5);

            if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
            {
                // Check momentum conservation
                // Momentum and energy conservation is not accurate (see the
                // DTMucfInteractor documentation for details). Thus, we only
                // check that the momentum calculation matches the
                // implementation and adds up to zero.
                auto const neutron_p_mag = this->calc_momentum(
                    sec[0].energy, data_.particle_masses.neutron);
                auto const muon_p_mag = this->calc_momentum(
                    sec[1].energy, data_.particle_masses.mu_minus);

                Real3 alpha_momentum, total_momentum;
                for (auto i : range(3))
                {
                    real_type neutron_momentum_i = sec[0].direction[i]
                                                   * neutron_p_mag;
                    real_type muon_momentum_i = sec[1].direction[i]
                                                * muon_p_mag;
                    alpha_momentum[i] = -(neutron_momentum_i + muon_momentum_i);
                    total_momentum[i] = neutron_momentum_i + muon_momentum_i
                                        + alpha_momentum[i];
                }

                EXPECT_VEC_SOFT_EQ(sec[2].direction,
                                   make_unit_vector(alpha_momentum));
                EXPECT_VEC_SOFT_EQ(Real3{}, total_momentum);
            }
        }

        else if (channel == Channel::muonicalpha_neutron)
        {
            ASSERT_EQ(num_secondaries_[channel],
                      interaction.secondaries.size());

            // Check particle types
            EXPECT_EQ(data_.particle_ids.neutron,
                      interaction.secondaries[0].particle_id);
            EXPECT_EQ(data_.particle_ids.muonic_alpha,
                      interaction.secondaries[1].particle_id);

            // Check directions are opposite
            EXPECT_SOFT_EQ(-1.0,
                           dot_product(interaction.secondaries[0].direction,
                                       interaction.secondaries[1].direction));
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
        3,  // alpha_muon_neutron
        2  // muonicalpha_neutron
    };
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(DTMucfInteractorTest, alpha_muon_neutron)
{
    auto const channel = DTMucfInteractor::Channel::alpha_muon_neutron;

    // Reserve space for 4 interactions with 3 secondaries each
    size_type const num_samples = 4;
    this->resize_secondaries(num_samples * num_secondaries_[channel]);
    auto& rng = this->rng();

    DTMucfInteractor interact(data_, channel, this->secondary_allocator());
    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto result = interact(rng);
        this->validate_interaction(result, channel);
    }
}

//---------------------------------------------------------------------------//
TEST_F(DTMucfInteractorTest, muonicalpha_neutron)
{
    auto const channel = DTMucfInteractor::Channel::muonicalpha_neutron;

    // Reserve space for 4 interactions with 2 secondaries each
    size_type const num_samples = 4;
    this->resize_secondaries(num_samples * num_secondaries_[channel]);
    auto& rng = this->rng();

    DTMucfInteractor interact(data_, channel, this->secondary_allocator());
    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto result = interact(rng);
        this->validate_interaction(result, channel);
    }
}

//---------------------------------------------------------------------------//
TEST_F(DTMucfInteractorTest, stress_test)
{
    size_type const num_samples = 10000;
    real_type total_avg_secondaries{0};

    for (auto channel : {DTMucfInteractor::Channel::alpha_muon_neutron,
                         DTMucfInteractor::Channel::muonicalpha_neutron})
    {
        this->resize_secondaries(num_samples * num_secondaries_[channel]);
        auto& rng = this->rng();

        DTMucfInteractor interact(data_, channel, this->secondary_allocator());
        for ([[maybe_unused]] auto i : range(num_samples))
        {
            auto result = interact(rng);
            total_avg_secondaries += result.secondaries.size();
        }
    }
    total_avg_secondaries /= 2 * num_samples;  // Average over both channels

    static real_type const expected_total_avg_secondaries{2.5};  // (3 + 2) / 2
    EXPECT_SOFT_EQ(expected_total_avg_secondaries, total_avg_secondaries);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
