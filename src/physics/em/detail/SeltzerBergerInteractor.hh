//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBergerInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Secondary.hh"
#include "base/StackAllocator.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"
#include "SeltzerBerger.hh"
#include "SBEnergyDistribution.hh"
#include "SBPositronXsCorrector.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer-Berger model for ...
 *
 */
class SeltzerBergerInteractor
{
  public:
    //!@{
    //! Type aliases
    using Energy   = units::MevEnergy;
    using EnergySq = SBEnergyDistribution::EnergySq;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    //! Construct sampler from shared and state data
    inline CELER_FUNCTION
    SeltzerBergerInteractor(const SeltzerBergerDeviceRef& device_pointers,
                            const ParticleTrackView&      particle,
                            const Real3&                  inc_direction,
                            const CutoffView&             cutoffs,
                            StackAllocator<Secondary>&    allocate,
                            const MaterialView&           material);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Bremsstrahlung gamma direction sampler from G4ModifiedTsai
    template<class Engine>
    inline CELER_FUNCTION real_type sample_cos_theta(Energy  kinetic_energy,
                                                     Engine& rng);

    // Device-side references
    const SeltzerBergerDeviceRef& device_pointers_;
    // Type of particle
    const ParticleId particle_id_;
    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle direction
    const Momentum inc_momentum_;
    // Interactor thresholds
    const CutoffView& cutoffs_;
    // Incident particle direction
    const Real3& inc_direction_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Material in which interaction occurs
    const MaterialView& material_;

    // Minimum energy for this interaction
    real_type energy_val_min_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "SeltzerBergerInteractor.i.hh"
