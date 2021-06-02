//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBergerInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/StackAllocator.hh"
#include "base/Types.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "SeltzerBerger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer-Berger model for electron and positron bremsstrahlung processes.
 *
 * Given an incoming electron or positron of sufficient energy (as per
 * CutOffView), this class provides the energy loss of these particles due to
 * radiation of photons in the field of a nucleus. This model improves accuracy
 * using cross sections based on interpolation of published tables from Seltzer
 * and Berger given in Nucl. Instr. and Meth. in Phys. Research B, 12(1):95–134
 * (1985) and Atomic Data and Nuclear Data Tables, 35():345 (1986). The cross
 * sections are obtained from SBEnergyDistribution and are appropriately scaled
 * in the case of positrons via SBPositronXsCorrector (to be done).
 *
 * \note This interactor performs an analogous sampling as in Geant4's
 * G4SeltzerBergerModel, documented in 10.2.1 of the Geant Physics Reference
 * (release 10.6). The implementation is based on Geant4 10.4.3.
 */
class SeltzerBergerInteractor
{
  public:
    //!@{
    //! Type aliases
    using Energy   = units::MevEnergy;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    //! Construct sampler from device/shared and state data
    inline CELER_FUNCTION
    SeltzerBergerInteractor(const SeltzerBergerNativeRef& shared,
                            const ParticleTrackView&      particle,
                            const Real3&                  inc_direction,
                            const CutoffView&             cutoffs,
                            StackAllocator<Secondary>&    allocate,
                            const MaterialView&           material,
                            const ElementComponentId&     elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Device (host CPU or GPU device) references
    const SeltzerBergerNativeRef& shared_;
    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle direction
    const Momentum inc_momentum_;
    // Incident particle direction
    const Real3& inc_direction_;
    // Incident particle flag for selecting XS correction factor
    const bool inc_particle_is_electron_;
    // Production cutoff for gammas
    const Energy gamma_cutoff_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Material in which interaction occurs
    const MaterialView& material_;
    // Element in which interaction occurs
    const ElementComponentId& elcomp_id_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "SeltzerBergerInteractor.i.hh"
