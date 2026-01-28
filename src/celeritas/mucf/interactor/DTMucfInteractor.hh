//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/interactor/DTMucfInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/distribution/IsotropicDistribution.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/mucf/data/DTMixMucfData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion of \f$ (dt)_\mu \f$ molecules.
 *
 * Fusion channels:
 * - \f$ \alpha + \mu + n \f$
 * - \f$ (\alpha)_\mu + n \f$
 *
 * \warning This implementation has an incorrect energy and momentum
 * conservation implementation. Acceleron assumes an isotropic direction for
 * both neutron and muon in the \f$ \alpha + \mu + n \f$ channel, which leads
 * to the alpha particle either conserving energy or momentum but not both
 * simultaneously. The current implementation results in a roughly correct
 * total energy within \f$ K_\text{total} = [17.5, 17.9] \f$ MeV, instead of
 * the expected 17.6 MeV.
 */
class DTMucfInteractor
{
  public:
    enum class Channel
    {
        alpha_muon_neutron,  //!< \f$ \alpha + \mu + n \f$
        muonicalpha_neutron,  //!< \f$ (\alpha)_\mu + n \f$
        size_
    };

    // Construct from shared and state data
    inline CELER_FUNCTION
    DTMucfInteractor(NativeCRef<DTMixMucfData> const& data,
                     Channel channel,
                     StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Shared constant physics properties
    NativeCRef<DTMixMucfData> const& data_;
    // Selected fusion channel
    Channel channel_{Channel::size_};
    // Allocate space for secondary particles
    StackAllocator<Secondary>& allocate_;

    // Number of secondaries per channel
    EnumArray<Channel, size_type> num_secondaries_{
        3,  // alpha_muon_neutron
        2  // muonicalpha_neutron
    };

    // Outgoing neutron kinetic energy
    inline CELER_FUNCTION units::MevEnergy neutron_kinetic_energy() const
    {
        return units::MevEnergy{14.1};
    }

    // Calculate momentum magnitude from energy and mass
    inline CELER_FUNCTION real_type calc_momentum(
        units::MevEnergy const energy, units::MevMass const mass) const;

    // Calculate kinetic energy from momentum and mass
    inline CELER_FUNCTION units::MevEnergy
    calc_kinetic_energy(Real3 const& momentum_vec,
                        units::MevMass const mass) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared data and channel selection.
 */
CELER_FUNCTION
DTMucfInteractor::DTMucfInteractor(NativeCRef<DTMixMucfData> const& data,
                                   Channel const channel,
                                   StackAllocator<Secondary>& allocate)
    : data_(data), channel_(channel), allocate_(allocate)
{
    CELER_EXPECT(data_);
    CELER_EXPECT(channel_ < Channel::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a dt muonic molecule fusion.
 */
template<class Engine>
CELER_FUNCTION Interaction DTMucfInteractor::operator()(Engine& rng)
{
    // Allocate space for the final fusion channel
    Secondary* sec = allocate_(num_secondaries_[channel_]);
    if (sec == nullptr)
    {
        // Failed to allocate space for secondaries
        return Interaction::from_failure();
    }

    size_type const neutron_idx{0};  // Both channels
    size_type const muon_idx{1}, alpha_idx{2};  // Channel::alpha_muon_neutron
    size_type const muonicalpha_idx{1};  // Channel::muonicalpha_neutron

    IsotropicDistribution sample_isotropic;
    // Grid range is [0,1), with its domain being the muon energy in MeV
    NonuniformGridCalculator sample_muon_energy(data_.muon_energy_cdf,
                                                data_.reals);

    // Neutron is the same on both cases: 14.1 MeV with random direction
    sec[neutron_idx].particle_id = data_.particle_ids.neutron;
    sec[neutron_idx].energy = this->neutron_kinetic_energy();
    sec[neutron_idx].direction = sample_isotropic(rng);

    switch (channel_)
    {
        case Channel::alpha_muon_neutron: {
            // Muon: random direction with energy sampled from its CDF
            sec[muon_idx].particle_id = data_.particle_ids.mu_minus;
            sec[muon_idx].direction = sample_isotropic(rng);
            sec[muon_idx].energy = units::MevEnergy{
                sample_muon_energy(generate_canonical(rng))};

            // Alpha: Final state calculated via momentum conservation
            sec[alpha_idx].particle_id = data_.particle_ids.alpha;

            // Momentum: p_alpha = - (p_neutron + p_muon)
            auto const neutron_momentum = this->calc_momentum(
                sec[neutron_idx].energy, data_.particle_masses.neutron);
            auto const muon_momentum = this->calc_momentum(
                sec[muon_idx].energy, data_.particle_masses.mu_minus);

            Real3 alpha_momentum_vec;
            for (auto i : range(3))
            {
                alpha_momentum_vec[i]
                    = -(sec[neutron_idx].direction[i] * neutron_momentum
                        + sec[muon_idx].direction[i] * muon_momentum);
            }
            sec[alpha_idx].direction = make_unit_vector(alpha_momentum_vec);

            // Energy: K = sqrt(p^2 + m^2) - m
            sec[alpha_idx].energy = this->calc_kinetic_energy(
                alpha_momentum_vec, data_.particle_masses.alpha);
            break;
        }

        case Channel::muonicalpha_neutron: {
            // Muonic alpha: Equal and opposite momentum to neutron
            sec[muonicalpha_idx].particle_id = data_.particle_ids.muonic_alpha;
            sec[muonicalpha_idx].energy = sec[neutron_idx].energy;
            for (auto i : range(3))
            {
                sec[muonicalpha_idx].direction[i]
                    = -sec[neutron_idx].direction[i];
            }
            break;
        }

        default:
            CELER_ASSERT_UNREACHABLE();
    }

    // Kill muon primary and generate fusion secondaries
    Interaction result = Interaction::from_absorption();
    result.secondaries = {sec, num_secondaries_[channel_]};
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate momentum magnitude from particle energy and mass via
 * \f$ p = \sqrt{K^2 + 2mK} \f$ .
 */
CELER_FUNCTION real_type DTMucfInteractor::calc_momentum(
    units::MevEnergy const energy, units::MevMass const mass) const

{
    return std::sqrt(ipow<2>(value_as<units::MevEnergy>(energy))
                     + 2 * value_as<units::MevMass>(mass)
                           * value_as<units::MevEnergy>(energy));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate kinetic energy given a particle's momentum and mass via
 * \f$ K = \sqrt{p^2 - m^2} - m\f$ .
 */
CELER_FUNCTION units::MevEnergy
DTMucfInteractor::calc_kinetic_energy(Real3 const& momentum_vec,
                                      units::MevMass const mass) const
{
    real_type momentum_mag = norm(momentum_vec);
    return units::MevEnergy{std::sqrt(ipow<2>(momentum_mag)
                                      + ipow<2>(value_as<units::MevMass>(mass)))
                            - value_as<units::MevMass>(mass)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
