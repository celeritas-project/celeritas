//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/MuPairProductionInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuPairProductionData.hh"
#include "celeritas/em/distribution/MuAngularDistribution.hh"
#include "celeritas/em/distribution/MuPPEnergyDistribution.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform electron-positron pair production by muons.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MuPairProductionModel and as documented in the Geant4 Physics Reference
 * Manual (Release 11.1) section 11.3.
 */
class MuPairProductionInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    MuPairProductionInteractor(NativeCRef<MuPairProductionData> const& shared,
                               ParticleTrackView const& particle,
                               CutoffView const& cutoffs,
                               ElementView const& element,
                               Real3 const& inc_direction,
                               StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// TYPES ////

    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using Momentum = units::MevMomentum;
    using UniformRealDist = UniformRealDistribution<real_type>;

    //// DATA ////

    // Shared model data
    NativeCRef<MuPairProductionData> const& shared_;
    // Allocate space for the secondary particle
    StackAllocator<Secondary>& allocate_;
    // Incident direction
    Real3 const& inc_direction_;
    // Incident particle energy [MeV]
    Energy inc_energy_;
    // Incident particle mass
    Mass inc_mass_;
    // Incident particle momentum [MeV / c]
    real_type inc_momentum_;
    // Sample the azimuthal angle
    UniformRealDist sample_phi_;
    // Sampler for the electron-positron pair energy
    MuPPEnergyDistribution sample_energy_;

    //// HELPER FUNCTIONS ////

    // Calculate the secondary particle momentum from the sampled energy
    inline CELER_FUNCTION real_type calc_momentum(Energy) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION MuPairProductionInteractor::MuPairProductionInteractor(
    NativeCRef<MuPairProductionData> const& shared,
    ParticleTrackView const& particle,
    CutoffView const& cutoffs,
    ElementView const& element,
    Real3 const& inc_direction,
    StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , allocate_(allocate)
    , inc_direction_(inc_direction)
    , inc_energy_(particle.energy())
    , inc_mass_(particle.mass())
    , inc_momentum_(value_as<Momentum>(particle.momentum()))
    , sample_phi_(0, real_type(2 * constants::pi))
    , sample_energy_(shared, particle, cutoffs, element)
{
    CELER_EXPECT(particle.particle_id() == shared.ids.mu_minus
                 || particle.particle_id() == shared.ids.mu_plus);
}

//---------------------------------------------------------------------------//
/*!
 * Simulate electron-positron pair production by muons.
 */
template<class Engine>
CELER_FUNCTION Interaction MuPairProductionInteractor::operator()(Engine& rng)
{
    // Allocate secondary electron and positron
    Secondary* secondaries = allocate_(2);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    // Sample the electron and positron energies
    auto energy = sample_energy_(rng);
    Energy pair_energy = energy.electron + energy.positron;

    // Sample the secondary directions
    MuAngularDistribution sample_costheta(inc_energy_, inc_mass_, pair_energy);
    real_type phi = sample_phi_(rng);

    // Create the secondary electron
    Secondary& electron = secondaries[0];
    electron.particle_id = shared_.ids.electron;
    electron.energy = energy.electron;
    electron.direction
        = rotate(from_spherical(sample_costheta(rng), phi), inc_direction_);

    // Create the secondary positron
    Secondary& positron = secondaries[1];
    positron.particle_id = shared_.ids.positron;
    positron.energy = energy.positron;
    positron.direction
        = rotate(from_spherical(sample_costheta(rng), phi + constants::pi),
                 inc_direction_);

    // Construct interaction for change to the incident muon
    Interaction result;
    result.secondaries = {secondaries, 2};
    result.energy = inc_energy_ - pair_energy;
    result.direction = make_unit_vector(
        inc_momentum_ * inc_direction_
        - this->calc_momentum(electron.energy) * electron.direction
        - this->calc_momentum(positron.energy) * positron.direction);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the secondary particle momentum from the sampled energy.
 */
CELER_FUNCTION real_type
MuPairProductionInteractor::calc_momentum(Energy energy) const
{
    return std::sqrt(ipow<2>(value_as<Energy>(energy))
                     + 2 * value_as<Mass>(shared_.electron_mass)
                           * value_as<Energy>(energy));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
