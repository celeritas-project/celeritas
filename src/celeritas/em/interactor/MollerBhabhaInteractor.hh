//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#include "detail/IoniFinalStateHelper.hh"

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
    Energy inc_energy_;
    // Incident momentum [MeV]
    Momentum inc_momentum_;
    // Incident direction
    Real3 const& inc_direction_;
    // Secondary electron cutoff for current material
    Energy electron_cutoff_;
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
    , inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_direction_(inc_direction)
    , electron_cutoff_(cutoffs.energy(shared_.ids.electron))
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
        /*!
         * \todo Remove and replace with an assertion once material-dependent
         * model bounds are supported
         */
        return Interaction::from_unchanged();
    }

    // Allocate memory for the produced electron
    Secondary* electron_secondary = allocate_(1);

    if (electron_secondary == nullptr)
    {
        // Fail to allocate space for a secondary
        return Interaction::from_failure();
    }

    // Sample secondary electron energy
    Energy secondary_energy = inc_energy_ * [this, &rng] {
        if (inc_particle_is_electron_)
        {
            return MollerEnergyDistribution(
                shared_.electron_mass, electron_cutoff_, inc_energy_)(rng);
        }
        return BhabhaEnergyDistribution(
            shared_.electron_mass, electron_cutoff_, inc_energy_)(rng);
    }();
    CELER_ASSERT(secondary_energy >= electron_cutoff_);

    // Sample the delta ray energy to construct the final sampler
    detail::IoniFinalStateHelper sample_interaction(inc_energy_,
                                                    inc_direction_,
                                                    inc_momentum_,
                                                    shared_.electron_mass,
                                                    secondary_energy,
                                                    shared_.electron_mass,
                                                    shared_.ids.electron,
                                                    electron_secondary);

    // Update kinematics of the final state and return this interaction
    return sample_interaction(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
