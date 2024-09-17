//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/interactor/MuDecayInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Constants.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/decay/data/MuDecayData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/InteractionUtils.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform muon decay.
 *
 */
class MuDecayInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    MuDecayInteractor(MuDecayData const& shared,
                      ParticleTrackView const& particle,
                      Real3 const& inc_direction,
                      StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Constant data
    MuDecayData shared_;
    // Incident muon energy
    units::MevEnergy const inc_energy_;
    // Incident direction
    Real3 const& inc_direction_;
    // Secondary electron or positron id
    ParticleId secondary_id_;
    // Allocate space for a secondary particles
    StackAllocator<Secondary>& allocate_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
MuDecayInteractor::MuDecayInteractor(MuDecayData const& shared,
                                     ParticleTrackView const& particle,
                                     Real3 const& inc_direction,
                                     StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.mu_minus
                 || particle.particle_id() == shared_.ids.mu_plus);

    secondary_id_ = (particle.particle_id() == shared_.ids.mu_minus)
                        ? shared_.ids.electron
                        : shared_.ids.positron;

    // TODO: Set up decay channel instead if we stop ignoring neutrinos
}

//---------------------------------------------------------------------------//
/*!
 * Sample a muon decay via the
 * \f[
 * \mu^- \longrightarrow e^- \overline{\nu}_e \nu_\mu
 * \f]
 * or
 * \f[
 * \mu^+ \longrightarrow e^+ \nu_e \overline{\nu}_\mu
 * \f]
 * channel, with a branching ratio of 100%.
 *
 * The decay is sampled at the muon's reference frame (i.e. at rest).
 */
template<class Engine>
CELER_FUNCTION Interaction MuDecayInteractor::operator()(Engine& rng)
{
    // Allocate space for the single electron or positron to be emitted
    // TODO: expand if we add neutrinos
    Secondary* charged_lepton = allocate_(1);
    if (charged_lepton == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    charged_lepton->particle_id = secondary_id_;

    UniformRealDistribution<real_type> sample_uniform;

    // TODO: This differs from physics manual, that states E_max = m_\mu / 2
    real_type max_electron_energy = real_type{0.5} * shared_.muon_mass
                                    - shared_.electron_mass;
    real_type x_max = real_type{1}
                      + ipow<2>(shared_.electron_mass)
                            / ipow<2>(shared_.muon_mass);

    real_type decay_rate;
    real_type electron_energy;
    real_type electron_neutrino_energy;
    real_type electron_energy_frac;
    real_type decay_rate_term;

    for ([[maybe_unused]] auto i : range(1000))
    {
        electron_energy_frac = sample_uniform(rng);

        for ([[maybe_unused]] auto j : range(1000))
        {
            electron_energy = max_electron_energy * sample_uniform(rng);
            decay_rate = sample_uniform(rng);
            if (decay_rate
                <= electron_energy * (real_type{1} - electron_energy))
            {
                break;
            }
            electron_energy = max_electron_energy;
        }
        electron_neutrino_energy = electron_energy;
        if (electron_neutrino_energy >= real_type{1} - electron_energy)
        {
            break;
        }
        electron_neutrino_energy = real_type{1} - electron_energy;
    }
    real_type muon_neutrino_energy = real_type{2} - electron_energy
                                     - electron_neutrino_energy;

    real_type cos_theta, sin_theta, r_phi, r_theta, r_psi;
    cos_theta = real_type{1} - 2 / electron_energy
                - 2 / electron_neutrino_energy
                + 2 / (electron_neutrino_energy * electron_energy);
    sin_theta = std::sqrt(real_type{1} - ipow<2>(cos_theta));

    real_type const twopi = real_type{2} * constants::pi;
    r_theta = std::acos(2 * sample_uniform(rng) - 1);
    r_phi = twopi * sample_uniform(rng);

    // Charged lepton
    charged_lepton->energy = std::sqrt(
        ipow<2>(electron_energy) * ipow<2>(max_electron_energy)
        + 2 * electron_energy * max_electron_energy * shared_.electron_mass);
    // Since it is at rest, a random initial direction is selected
    charged_lepton->direction = {0, 0, 1}
                                * rotate(from_spherical(r_theta, r_phi));

    // Temporary neutrinos; not added to the final decay result

    // Electron neutrino
    Secondary electron_neutrino;

    // The extra term becomes necessary if neutrino masses are set as non-zero
    electron_neutrino.energy
        = sqrt(ipow<2>(electron_neutrino_energy) * ipow<2>(max_electron_energy)
               /* + 2 * electron_neutrino_energy * max_electron_energy * electron_neutrino_mass */);
    electron_neutrino.direction = {sin_theta, 0, cos_theta}
                                  * rotate(from_spherical(r_theta, r_phi));

    // Muon neutrino
    Secondary muon_neutrino;

    // The extra term becomes necessary if neutrino masses are set as non-zero
    muon_neutrino.energy
        = sqrt(ipow<2>(muon_neutrino_energy) * ipow<2>(max_electron_energy)
               /* + 2 * muon_neutrino_energy * max_electron_energy * muon_neutrino_mass */);
    muon_neutrino.direction
        = {-sin_theta * electron_neutrino_energy / muon_neutrino_energy,
           0,
           -electron_energy / muon_neutrino_energy
               - cos_theta * electron_neutrino_energy / muon_neutrino_energy}
          * rotate(from_spherical(r_theta, r_phi));

    Interaction result;
    result.action = Interaction::Action::decay;
    result.secondaries = {charged_lepton, 1};  // TODO: expand if we add nu's

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
