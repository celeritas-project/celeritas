//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/interactor/DDMucfInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/StackAllocator.hh"
#include "celeritas/mucf/data/DTMixMucfData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/Secondary.hh"

#include "detail/MucfInteractorUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion of \f$ (dd)_\mu \f$ molecules.
 *
 * Fusion channels:
 * - \f$ ^3\text{He} + \mu + n \f$
 * - \f$ (^3\text{He})_\mu + n \f$
 * - \f$ t + \mu + p \f$
 *
 * The mass ratios between \f$ ^3\text{He} \f$ and the neutron, and between
 * tritium and the proton, are both about 3:1. This leads to the neutron and
 * proton kinetic energies being 3/4 of the total kinetic energy available in
 * their respective channels.
 *
 * \note This interactor has a similar implementation as \c DTMucfInteractor ,
 * where energy is not fully conserved. See its documentation for details.
 */
class DDMucfInteractor
{
  public:
    //! \todo Implement muonictriton_proton (\f$ (t)_\mu + p \f$)
    enum class Channel
    {
        helium3_muon_neutron,  //!< \f$ ^3\text{He} + \mu + n \f$
        muonichelium3_neutron,  //!< \f$ (^3\text{He})_\mu + n \f$
        tritium_muon_proton,  //!< \f$ t + \mu + p \f$
        size_
    };

    // Construct from shared and state data
    inline CELER_FUNCTION
    DDMucfInteractor(NativeCRef<DTMixMucfData> const& data,
                     Channel channel,
                     StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Shared constant physics properties
    NativeCRef<DTMixMucfData> const& data_;
    // Selected fusion channel
    Channel channel_;
    // Allocate space for secondary particles
    StackAllocator<Secondary>& allocate_;
    // Number of secondaries per channel
    EnumArray<Channel, size_type> num_secondaries_{
        3,  // helium3_muon_neutron
        2,  // muonichelium3_neutron
        3  // tritium_muon_proton
    };

    // Neutron channels total kinetic energy
    inline CELER_FUNCTION units::MevEnergy total_energy_neutron_channels() const
    {
        return units::MevEnergy{3.3};
    }

    // Proton channel total kinetic energy
    inline CELER_FUNCTION units::MevEnergy total_energy_proton_channel() const
    {
        return units::MevEnergy{4.03};
    }

    // Outgoing neutron kinetic energy
    inline CELER_FUNCTION units::MevEnergy neutron_kinetic_energy() const
    {
        return 0.75 * this->total_energy_neutron_channels();
    }

    // Outgoing proton kinetic energy
    inline CELER_FUNCTION units::MevEnergy proton_kinetic_energy() const
    {
        return 0.75 * this->total_energy_proton_channel();
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
DDMucfInteractor::DDMucfInteractor(NativeCRef<DTMixMucfData> const& data,
                                   Channel channel,
                                   StackAllocator<Secondary>& allocate)
    : data_(data), channel_(channel), allocate_(allocate)
{
    CELER_EXPECT(data_);
    CELER_EXPECT(channel < Channel::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a dd muonic molecule fusion.
 */
template<class Engine>
CELER_FUNCTION Interaction DDMucfInteractor::operator()(Engine& rng)
{
    // Allocate space for the final fusion channel
    Secondary* sec = allocate_(num_secondaries_[channel_]);
    if (sec == nullptr)
    {
        // Failed to allocate space for secondaries
        return Interaction::from_failure();
    }

    // Short aliases for secondary indices
    size_type const neutron_idx{0}, proton_idx{0};
    size_type const muon_idx{1}, muonic_he3{1};
    size_type const he3_idx{2}, tritium_idx{2};

    IsotropicDistribution sample_isotropic;

    switch (channel_)
    {
        case Channel::helium3_muon_neutron: {
            // Neutron: random direction with known energy
            sec[neutron_idx] = detail::sample_mucf_secondary(
                data_.particle_ids.neutron, this->neutron_kinetic_energy(), rng);

            // Muon: random direction with energy sampled from its CDF
            sec[muon_idx] = detail::sample_mucf_muon(
                data_.particle_ids.mu_minus,
                NonuniformGridCalculator{data_.muon_energy_cdf, data_.reals},
                rng);

            // Helium-3: momentum conservation
            sec[he3_idx]
                = detail::calc_third_secondary(sec[neutron_idx],
                                               data_.particle_masses.neutron,
                                               sec[muon_idx],
                                               data_.particle_masses.mu_minus,
                                               data_.particle_ids.he3,
                                               data_.particle_masses.he3);
            break;
        }

        case Channel::muonichelium3_neutron: {
            // Neutron: random direction with known energy
            sec[neutron_idx] = detail::sample_mucf_secondary(
                data_.particle_ids.neutron, this->neutron_kinetic_energy(), rng);

            // Muonic helium-3: momentum conservation
            sec[muonic_he3].particle_id = data_.particle_ids.muonic_he3;
            sec[muonic_he3].energy = this->total_energy_neutron_channels()
                                     - this->neutron_kinetic_energy();
            sec[muonic_he3].direction
                = detail::opposite(sec[neutron_idx].direction);
            break;
        }

        case Channel::tritium_muon_proton: {
            // Proton: random direction with known energy
            sec[proton_idx] = detail::sample_mucf_secondary(
                data_.particle_ids.proton, this->proton_kinetic_energy(), rng);

            // Muon: random direction with energy sampled from its CDF
            sec[muon_idx] = detail::sample_mucf_muon(
                data_.particle_ids.mu_minus,
                NonuniformGridCalculator{data_.muon_energy_cdf, data_.reals},
                rng);

            // tritium: momentum conservation
            sec[tritium_idx]
                = detail::calc_third_secondary(sec[proton_idx],
                                               data_.particle_masses.proton,
                                               sec[muon_idx],
                                               data_.particle_masses.mu_minus,
                                               data_.particle_ids.triton,
                                               data_.particle_masses.triton);
            break;
        }

        default:
            CELER_ASSERT_UNREACHABLE();
    }

    // Kill primary and generate secondaries
    Interaction result = Interaction::from_absorption();
    result.secondaries = {sec, num_secondaries_[channel_]};
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
