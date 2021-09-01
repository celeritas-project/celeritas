//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Secondary.hh"
#include "base/StackAllocator.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating a BetheHeitlerInteractor.
 */
struct BetheHeitlerPointers
{
    //! Model ID
    ModelId model_id;

    //! Inverse of electron mass [1 / MevMass]
    real_type inv_electron_mass;
    //! ID of an electron
    ParticleId electron_id;
    //! ID of an positron
    ParticleId positron_id;
    //! ID of a gamma
    ParticleId gamma_id;

    //! Check whether the view is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model_id && inv_electron_mass > 0 && electron_id && positron_id
               && gamma_id;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Bethe-Heitler model for gamma -> e+e- (electron-pair production).
 *
 * Give an incident gamma, it adds a two pair-produced secondary electrons to
 * the secondary stack. No cutoffs are performed on the incident gamma energy.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4BetheHeitlerModel, as documented in section 6.5 of the Geant4 Physics
 * Reference (release 10.6), applicable to incident gammas with energy
 * \f$ E_gamma \leq 100\f$ GeV . For \f$ E_gamma > 80 \f$ GeV, it is suggested
 * to use `G4PairProductionRelModel`.
 */
class BetheHeitlerInteractor
{
  public:
    //!@{
    //! Type aliases
    using MevMass = units::MevMass;
    //!@}

  public:
    //! Construct sampler from shared and state data
    inline CELER_FUNCTION
    BetheHeitlerInteractor(const BetheHeitlerPointers& shared,
                           const ParticleTrackView&    particle,
                           const Real3&                inc_direction,
                           StackAllocator<Secondary>&  allocate,
                           const ElementView&          element);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Gamma energy divided by electron mass * csquared
    const BetheHeitlerPointers& shared_;
    // Incident gamma energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Element properties for calculating screening functions and variables
    const ElementView& element_;
    // Cached minimum epsilon, m_e*c^2/E_gamma; kinematical limit for Y -> e+e-
    real_type epsilon0_;

    //// HELPER FUNCTIONS ////

    // Calculates the screening variable, \$f \delta \eta \$f, which is a
    // function of \$f \epsilon \$f. This is a measure of the "impact
    // parameter" of the incident photon.
    inline CELER_FUNCTION real_type impact_parameter(real_type eps) const;

    // Screening function, Phi_1, for the corrected Bethe-Heitler
    // cross-section calculation.
    inline CELER_FUNCTION real_type
    screening_phi1(real_type impact_parameter) const;

    // Screening function, Phi_2, for the corrected Bethe-Heitler
    // cross-section calculation.
    inline CELER_FUNCTION real_type
    screening_phi2(real_type impact_parameter) const;

    // Auxiliary screening function, Phi_1, for the "composition+rejection"
    // technique for sampling.
    inline CELER_FUNCTION real_type screening_phi1_aux(real_type delta) const;

    // Auxiliary screening function, Phi_2, for the "composition+rejection"
    // technique for sampling.
    inline CELER_FUNCTION real_type screening_phi2_aux(real_type delta) const;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "BetheHeitlerInteractor.i.hh"
