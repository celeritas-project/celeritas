//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/MollerBhabhaInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MollerBhabhaData.hh"
#include "celeritas/em/distribution/BhabhaEnergyDistribution.hh"
#include "celeritas/em/distribution/MollerEnergyDistribution.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform Moller (e-e-) and Bhabha (e+e-) scattering.
 *
 * This interaction, part of the ionization process, is when an incident
 * electron or positron ejects an electron from the surrounding matter.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MollerBhabhaModel class, as documented in section 10.1.4 of the Geant4
 * Physics Reference (release 10.6).
 */
class MollerBhabhaInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Mass = units::MevMass;
    using Energy = units::MevEnergy;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    //! Construct with shared and state data
    inline CELER_FUNCTION
    MollerBhabhaInteractor(MollerBhabhaData const& shared,
                           ParticleTrackView const& particle,
                           CutoffView const& cutoffs,
                           Real3 const& inc_direction,
                           StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Shared constant physics properties
    MollerBhabhaData const& shared_;
    // Incident energy [MeV]
    const real_type inc_energy_;
    // Incident momentum [MeV]
    const real_type inc_momentum_;
    // Incident direction
    Real3 const& inc_direction_;
    // Secondary electron cutoff for current material
    const real_type electron_cutoff_;
    // Allocate space for the secondary particle
    StackAllocator<Secondary>& allocate_;
    // Incident particle flag for selecting Moller or Bhabha scattering
    bool const inc_particle_is_electron_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The incident particle must be within the model's valid energy range. this
 * must be handled in code *before* the interactor is constructed.
 */
CELER_FUNCTION MollerBhabhaInteractor::MollerBhabhaInteractor(
    MollerBhabhaData const& shared,
    ParticleTrackView const& particle,
    CutoffView const& cutoffs,
    Real3 const& inc_direction,
    StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , inc_energy_{value_as<Energy>(particle.energy())}
    , inc_momentum_{value_as<Momentum>(particle.momentum())}
    , inc_direction_(inc_direction)
    , electron_cutoff_(value_as<Energy>(cutoffs.energy(shared_.ids.electron)))
    , allocate_(allocate)
    , inc_particle_is_electron_(particle.particle_id() == shared_.ids.electron)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.electron
                 || particle.particle_id() == shared_.ids.positron);
}

//---------------------------------------------------------------------------//
/*!
 * Sample e-e- or e+e- scattering using Moller or Bhabha models, depending on
 * the incident particle.
 *
 * See section 10.1.4 of the Geant4 physics reference manual (release 10.6).
 */
template<class Engine>
CELER_FUNCTION Interaction MollerBhabhaInteractor::operator()(Engine& rng)
{
    if (inc_energy_ <= (inc_particle_is_electron_ ? 2 : 1) * electron_cutoff_)
    {
        // The secondary should not be emitted. This interaction cannot
        // happen and the incident particle must undergo an energy loss
        // process.
        return Interaction::from_unchanged(Energy{inc_energy_}, inc_direction_);
    }

    // Allocate memory for the produced electron
    Secondary* electron_secondary = this->allocate_(1);

    if (electron_secondary == nullptr)
    {
        // Fail to allocate space for a secondary
        return Interaction::from_failure();
    }

    // Sample energy transfer fraction
    real_type epsilon;
    if (inc_particle_is_electron_)
    {
        MollerEnergyDistribution sample_moller(
            value_as<Mass>(shared_.electron_mass),
            electron_cutoff_,
            inc_energy_);
        epsilon = sample_moller(rng);
    }
    else
    {
        BhabhaEnergyDistribution sample_bhabha(
            value_as<Mass>(shared_.electron_mass),
            electron_cutoff_,
            inc_energy_);
        epsilon = sample_bhabha(rng);
    }

    // Sampled secondary kinetic energy
    const real_type secondary_energy = epsilon * inc_energy_;
    CELER_ASSERT(secondary_energy >= electron_cutoff_);

    // Same equation as in ParticleTrackView::momentum_sq()
    // TODO: use local data ParticleTrackView
    const real_type secondary_momentum = std::sqrt(
        secondary_energy
        * (secondary_energy + 2 * value_as<Mass>(shared_.electron_mass)));

    const real_type total_energy = inc_energy_
                                   + value_as<Mass>(shared_.electron_mass);

    // Calculate theta from energy-momentum conservation
    real_type secondary_cos_theta
        = secondary_energy
          * (total_energy + value_as<Mass>(shared_.electron_mass))
          / (secondary_momentum * inc_momentum_);

    secondary_cos_theta = celeritas::min<real_type>(secondary_cos_theta, 1);
    CELER_ASSERT(secondary_cos_theta >= -1 && secondary_cos_theta <= 1);

    // Sample phi isotropically
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);

    // Create cartesian direction from the sampled theta and phi
    Real3 secondary_direction = rotate(
        from_spherical(secondary_cos_theta, sample_phi(rng)), inc_direction_);

    // Calculate incident particle final direction
    Real3 inc_exiting_direction;
    for (int i = 0; i < 3; ++i)
    {
        real_type inc_momentum_ijk = inc_momentum_ * inc_direction_[i];
        real_type secondary_momentum_ijk = secondary_momentum
                                           * secondary_direction[i];
        inc_exiting_direction[i] = inc_momentum_ijk - secondary_momentum_ijk;
    }
    normalize_direction(&inc_exiting_direction);

    // Construct interaction for change to primary (incident) particle
    const real_type inc_exiting_energy = inc_energy_ - secondary_energy;
    Interaction result;
    result.energy = Energy{inc_exiting_energy};
    result.secondaries = {electron_secondary, 1};
    result.direction = inc_exiting_direction;

    // Assign values to the secondary particle
    electron_secondary[0].particle_id = shared_.ids.electron;
    electron_secondary[0].energy = Energy{secondary_energy};
    electron_secondary[0].direction = secondary_direction;

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
