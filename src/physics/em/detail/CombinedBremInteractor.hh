//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CombinedBremInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "base/StackAllocator.hh"

#include "physics/base/CutoffView.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"

#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "physics/material/Types.hh"

#include "CombinedBremData.hh"
#include "RelativisticBremDXsection.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * This is a combination of the Seltzer-Berger model and the relativistic
 * bremsstrahlung model for electrons and positorons bremsstrahlung processes.
 *
 */
class CombinedBremInteractor
{
    //!@{
    //! Type aliases
    using Energy      = units::MevEnergy;
    using Momentum    = units::MevMomentum;
    using ElementData = detail::RelBremElementData;
    using ItemIdT     = celeritas::ItemId<unsigned int>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    CombinedBremInteractor(const CombinedBremNativeRef& shared,
                           const ParticleTrackView&     particle,
                           const Real3&                 direction,
                           const CutoffView&            cutoffs,
                           StackAllocator<Secondary>&   allocate,
                           const MaterialView&          material,
                           const ElementComponentId&    elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant physics properties
    const CombinedBremNativeRef& shared_;
    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle direction
    const Momentum inc_momentum_;
    // Incident particle direction
    const Real3& inc_direction_;
    // Production cutoff for gammas
    const Energy gamma_cutoff_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Material in which interaction occurs
    const MaterialView& material_;
    // Element in which interaction occurs
    const ElementComponentId elcomp_id_;
    // Differential cross section calcuator for the relativistic interactor
    RelativisticBremDXsection rb_dxsec_;
    // Incident particle flag for selecting XS correction factor
    const bool is_electron_;
    // Flag for selecting the relativistic bremsstrahlung model
    const bool is_relativistic_;

    //// HELPER FUNCTIONS ////

    //! Sample the bremsstrahlung energy at the low energy (SeltzerBerger)
    template<class Engine>
    inline CELER_FUNCTION Energy sample_energy_sb(Engine& rng);

    //! Sample the bremsstrahlung energy by the high energy (RelativisticBrem)
    template<class Engine>
    inline CELER_FUNCTION Energy sample_energy_rb(Engine& rng);

    //! Update the final state after the interaction
    template<class Engine>
    inline CELER_FUNCTION Interaction update_state(Engine&      rng,
                                                   const Energy energy,
                                                   Secondary*   secondaries);
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "CombinedBremInteractor.i.hh"
