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
#include "corecel/math/Algorithms.hh"
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
    // \todo Expand if we add neutrinos
    Secondary* charged_lepton = allocate_(1);
    if (charged_lepton == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    UniformRealDistribution<real_type> sample_uniform;

    // \todo This differs from physics manual, that states E_max = m_\mu / 2
    real_type max_electron_energy = real_type{0.5} * shared_.muon_mass
                                    - shared_.electron_mass;
    real_type x_max = real_type{1}
                      + ipow<2>(shared_.electron_mass)
                            / ipow<2>(shared_.muon_mass);

    real_type decay_rate;
    real_type electron_energy;
    real_type electron_neutrino_energy_frac;
    real_type electron_energy_frac;

    size_type const max_iter = 1000;  // \todo Why?
    size_type outer_loop{0}, inner_loop{0};

    do
    {
        outer_loop++;
        electron_energy_frac = sample_uniform(rng);
        do
        {
            inner_loop++;
            electron_energy = max_electron_energy * sample_uniform(rng);
            decay_rate = sample_uniform(rng);

            if (inner_loop > max_iter)
            {
                electron_energy = max_electron_energy;
                break;
            }

        } while (decay_rate
                 > electron_energy * (real_type{1} - electron_energy));

        if (outer_loop > max_iter)
        {
            electron_neutrino_energy_frac = 1 - electron_energy_frac;
            break;
        }
    } while (electron_neutrino_energy_frac
             < real_type{1} - electron_energy_frac);

    real_type muon_neutrino_energy = real_type{2} - electron_energy_frac
                                     - electron_neutrino_energy_frac;

    // Angle between charged lepton and electron neutrino
    real_type cos_theta
        = real_type{1} - 2 / electron_energy - 2 / electron_neutrino_energy_frac
          + 2 / (electron_neutrino_energy_frac * electron_energy);
    CELER_ASSERT(std::fabs(cos_theta) <= 1);

    real_type sin_theta = std::sqrt(1 - ipow<2>(cos_theta));
    CELER_ASSERT(std::fabs(sin_theta) <= 1);

    // Sample spherically uniform direction of the charged lepton
    real_type sampled_theta = 2 * constants::pi * sample_uniform(rng);
    real_type sampled_phi = constants::pi * sample_uniform(rng);

    //// Charged lepton ////
    charged_lepton->particle_id = secondary_id_;
    charged_lepton->energy = std::sqrt(
        ipow<2>(electron_energy) * ipow<2>(max_electron_energy)
        + 2 * electron_energy * max_electron_energy * shared_.electron_mass);
    charged_lepton->direction
        = rotate({0, 0, 1}, from_spherical(sampled_theta, sampled_phi));

    //// Electron neutrino ////
    Secondary* electron_neutrino;  // Not added to the decay result

    // Extra term is necessary if neutrino masses are not neglected
    electron_neutrino->energy
        = sqrt(ipow<2>(electron_neutrino_energy_frac) * ipow<2>(max_electron_energy)
               /* + 2 * electron_neutrino_energy * max_electron_energy * electron_neutrino_mass */);
    electron_neutrino->direction = rotate(
        {sin_theta, 0, cos_theta}, from_spherical(sampled_theta, sampled_phi));

    //// Muon neutrino ////
    Secondary* muon_neutrino;  // Not added to the decay result

    // Extra term is necessary if neutrino masses are not neglected
    muon_neutrino->energy
        = sqrt(ipow<2>(muon_neutrino_energy) * ipow<2>(max_electron_energy)
               /* + 2 * muon_neutrino_energy * max_electron_energy * muon_neutrino_mass */);
    muon_neutrino->direction = rotate(
        {-sin_theta * electron_neutrino_energy_frac / muon_neutrino_energy,
         0,
         -electron_energy / muon_neutrino_energy
             - cos_theta * electron_neutrino_energy_frac / muon_neutrino_energy},
        from_spherical(theta, phi));

    Interaction result;
    result.action = Interaction::Action::decay;
    result.secondaries = {charged_lepton, 1};  // \todo Expand if we add nu's

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
