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
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/base/Units.hh"
#include "BetheHeitlerInteractorPointers.hh"

#include "Material.mock.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Bethe-Heitler model for gamma -> e+e- (electron-pair production).
 *
 * Give an incident gamma, it adds a two pair-produced secondary electrons to
 * the secondary stack. No cutoffs are performed on the incident gamma energy.
 *
 * \note This performs the same sampling routine as in Geant4's
 *  G4BetheHeitlerModel, as documented in section 6.5 of the Geant4 Physics
 *  Reference (release 10.6), applicable to incident gammas with energy
 *  E_gamma \leq 100 GeV. For E_gamma \gt 100 GeV, the cross section is
 * constant.
 */
class BetheHeitlerInteractor
{
  public:
    //! Construct sampler from shared and state data
    //! TODO: Handle Material through `shared`?
    inline CELER_FUNCTION
    BetheHeitlerInteractor(const BetheHeitlerInteractorPointers& shared,
                           const ParticleTrackView&              particle,
                           const Real3&                          inc_direction,
                           SecondaryAllocatorView&               allocate,
                           const MaterialMock&                   material);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

    // >>> COMMON PROPERTIES

    // Minimum incident gamma energy for this model
    // (used for the parameterization in the
    // cross-section calculation).
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {
        return units::MevEnergy{1.5}; // 1.5 MeV
    }

    // Maximum incident gamma energy for this mode (used for the
    // parameterization in the cross-section calculation). Above this energy,
    // the cross section is constant.
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {
        return units::MevEnergy{100000.0}; // 100 GeV
    }

  private:
    // Calculates the screening variable, \deta, which is a function of
    // \epsilon. This is a measure of the "impact parameter" of the incident
    // photon.
    inline CELER_FUNCTION real_type delta(size_type Z, real_type eps) const;

    // Screening function, Phi1(), for the corrected Bethe-Heitler
    // cross-section calculation.
    inline CELER_FUNCTION real_type Phi1(real_type delta) const;

    // Screening function, Phi2(), for the corrected Bethe-Heitler
    // cross-section calculation.
    inline CELER_FUNCTION real_type Phi2(real_type delta) const;

    // Screening function for the case that the screening variable \delta > 1;
    // in this case, \Phi1(\delta) = \Phi2(\delta) = \Phi12(delta).
    inline CELER_FUNCTION real_type Phi12(real_type delta) const;

    // Born approximation -- Coulomb correction function, F(Z), instead of
    // plane waves. When E_gamma >= 50 MeV, calculated to fourth-order in the
    // fine-structure constant, \alpha.
    inline CELER_FUNCTION real_type CoulombCorr(size_type Z) const;

    // "Auxiliary" Coulomb correction function.
    inline CELER_FUNCTION real_type CoulombCorr_aux(size_type Z) const;

    // Auxiliary screening function, Phi1, for the "composition+rejection"
    // technique for sampling.
    inline CELER_FUNCTION real_type Phi1_aux(real_type delta,
                                             size_type Z) const;

    // Auxiliary screening function, Phi2, for the "composition+rejection"
    // technique for sampling.
    inline CELER_FUNCTION real_type Phi2_aux(real_type delta,
                                             size_type Z) const;

    // Density function for sampling the polar angle of the electron/positron.
    // The angle is defiend with respect to the direction of the parent photon.
    // inline CELER_FUNCTION real_type polar_angle_density() const;

    // Gamma energy divided by electron mass * csquared
    const BetheHeitlerInteractorPointers& shared_;

    // Incident gamma energy
    const units::MevEnergy inc_energy_;

    // Incident direction
    const Real3& inc_direction_;

    // Allocate space for a secondary particle
    SecondaryAllocatorView& allocate_;

    // Material atomic number, Z, needed for screening functions and variables
    const MaterialMock& material_;

    // Cached minimum epsilon, m_e*c^2/E_gamma; kinematical limit for Y -> e+e-
    real_type epsilon0_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "BetheHeitlerInteractor.i.hh"
