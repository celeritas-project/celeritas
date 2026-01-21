//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/DTMucfInteractor.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/mucf/interactor/DTMucfInteractor.hh"

#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/NonuniformGridBuilder.hh"
#include "celeritas/inp/MucfPhysics.hh"
#include "celeritas/phys/InteractionIO.hh"

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
        auto const& params = *this->particle_params();
        this->set_material("hdt_fuel");

        HostVal<DTMixMucfData> host_data;

        // Set up particle IDs
        host_data.particle_ids.mu_minus = params.find(pdg::mu_minus());
        host_data.particle_ids.neutron = params.find(pdg::neutron());
        host_data.particle_ids.alpha = params.find(pdg::alpha());
        host_data.particle_ids.muonic_alpha = params.find(pdg::muonic_alpha());

        // Set up particle masses
        host_data.particle_masses.mu_minus
            = params.get(host_data.particle_ids.mu_minus).mass();
        host_data.particle_masses.neutron
            = params.get(host_data.particle_ids.neutron).mass();
        host_data.particle_masses.alpha
            = params.get(host_data.particle_ids.alpha).mass();
        host_data.particle_masses.muonic_alpha
            = params.get(host_data.particle_ids.muonic_alpha).mass();

        // Set up muon energy CDF
        auto const inp_data = inp::MucfPhysics::from_default();
        NonuniformGridBuilder build_grid_record{&host_data.reals};
        host_data.muon_energy_cdf = build_grid_record(inp_data.muon_energy_cdf);

        // Construct collection
        data_ = CollectionMirror<DTMixMucfData>{std::move(host_data)};

        // At-rest muon primary
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{0.0});
        this->set_inc_direction({1, 0, 0});
    }

    // Detailed validation of the interaction result
    void
    validate_interaction(Interaction const& interaction, Channel channel) const
    {
        auto host_data = data_.host_ref();

        // Primary muon should be killed
        EXPECT_EQ(Action::absorbed, interaction.action);

        // First particle is always an outgoing neutron with 14.1 MeV
        EXPECT_EQ(host_data.particle_ids.neutron,
                  interaction.secondaries[0].particle_id);
        EXPECT_SOFT_EQ(14.1, interaction.secondaries[0].energy.value());

        // Verify channel-specific data
        if (channel == Channel::alpha_muon_neutron)
        {
            ASSERT_EQ(num_secondaries_[channel],
                      interaction.secondaries.size());

            // Check particles
            EXPECT_EQ(host_data.particle_ids.mu_minus,
                      interaction.secondaries[1].particle_id);
            EXPECT_EQ(host_data.particle_ids.alpha,
                      interaction.secondaries[2].particle_id);

            // Check energy conservation (17.6 MeV total)
            real_type total_kinetic_energy = 0;
            for (auto const& sec : interaction.secondaries)
            {
                total_kinetic_energy += sec.energy.value();
            }
            EXPECT_SOFT_EQ(17.6, total_kinetic_energy);

            // Check momentum conservation (total momentum must be zero)
            auto const neutron_p_mag
                = this->calc_momentum(interaction.secondaries[0].energy,
                                      host_data.particle_masses.neutron);
            auto const muon_p_mag
                = this->calc_momentum(interaction.secondaries[1].energy,
                                      host_data.particle_masses.mu_minus);
            auto const alpha_p_mag
                = this->calc_momentum(interaction.secondaries[2].energy,
                                      host_data.particle_masses.alpha);

            Real3 total_momentum;
            for (int i = 0; i < 3; ++i)
            {
                total_momentum[i]
                    = interaction.secondaries[0].direction[i] * neutron_p_mag
                      + interaction.secondaries[1].direction[i] * muon_p_mag
                      + interaction.secondaries[2].direction[i] * alpha_p_mag;
            }
            EXPECT_VEC_SOFT_EQ(Real3{}, total_momentum);
        }

        else if (channel == Channel::muonicalpha_neutron)
        {
            ASSERT_EQ(num_secondaries_[channel],
                      interaction.secondaries.size());

            // Check particle types
            EXPECT_EQ(host_data.particle_ids.neutron,
                      interaction.secondaries[0].particle_id);
            EXPECT_EQ(host_data.particle_ids.muonic_alpha,
                      interaction.secondaries[1].particle_id);

            // Check kinetic energies are equal
            EXPECT_SOFT_EQ(interaction.secondaries[0].energy.value(),
                           interaction.secondaries[1].energy.value());

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
    CollectionMirror<DTMixMucfData> data_;
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
    int const num_samples = 1;
    this->resize_secondaries(3 * num_samples);

    // Run interactor
    DTMucfInteractor interact(
        data_.host_ref(), channel, this->secondary_allocator());

    auto& rng = this->rng();
    for (auto i : range(num_samples))
    {
        Interaction result = interact(rng);
        this->validate_interaction(result, channel);
    }
}

//---------------------------------------------------------------------------//
TEST_F(DTMucfInteractorTest, muonicalpha_neutron)
{
    auto const channel = DTMucfInteractor::Channel::muonicalpha_neutron;

    // Reserve space for 4 interactions with 2 secondaries each
    int const num_samples = 4;
    this->resize_secondaries(2 * num_samples);

    // Run interactor
    DTMucfInteractor interact(
        data_.host_ref(), channel, this->secondary_allocator());

    auto& rng = this->rng();
    for (auto i : range(num_samples))
    {
        Interaction result = interact(rng);
        this->validate_interaction(result, channel);
    }
}

//---------------------------------------------------------------------------//
TEST_F(DTMucfInteractorTest, stress_test)
{
    size_type const num_samples = 1000;

    real_type total_avg_secondaries{0};

    for (auto channel : {DTMucfInteractor::Channel::alpha_muon_neutron,
                         DTMucfInteractor::Channel::muonicalpha_neutron})
    {
        this->resize_secondaries(num_samples * num_secondaries_[channel]);

        DTMucfInteractor interact(
            data_.host_ref(), channel, this->secondary_allocator());

        auto& rng = this->rng();
        for (auto i : range(num_samples))
        {
            Interaction result = interact(rng);
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
