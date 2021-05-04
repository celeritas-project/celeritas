//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"
#include "physics/material/Types.hh"
#include "Rayleigh.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * This is a model for the Rayleigh scattering process for photons.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4LivermoreRayleighModel class, as documented in section 6.2.2 of the
 * Geant4 Physics Reference (release 10.6).
 */
class RayleighInteractor
{
    //!@{
    //! Type aliases
    using ItemIdT = celeritas::ItemId<unsigned int>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION RayleighInteractor(const RayleighNativeRef& shared,
                                             const ParticleTrackView& particle,
                                             const Real3& inc_direction,
                                             ElementId    element_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

    //// COMMON PROPERTIES ////

    //! Minimum incident energy for this model to be valid: 10 * eV
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {
        return units::MevEnergy{1.0e-5};
    }

    //! Maximum incident energy for this model to be valid: 1 * GeV
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {
        return units::MevEnergy{1.0e+3};
    }

  private:
    //! cm/hc in the MeV energy unit
    static CELER_CONSTEXPR_FUNCTION real_type hc_factor()
    {
        return units::centimeter * unit_cast(units::MevEnergy{1.0})
               / (constants::c_light * constants::h_planck);
    }

    //! A point where the functional form of the form factor fit changes
    static CELER_CONSTEXPR_FUNCTION real_type fit_slice() { return 0.02; }

    //! Intermediate data for sampling input
    struct SampleInput
    {
        real_type factor{0};
        Real3     weight{0, 0, 0};
        Real3     prob{0, 0, 0};
    };

    //! Evaluate weights and probabilities for the angular sampling algorithm
    CELER_FUNCTION auto evaluate_weight_and_prob() const -> SampleInput;

  private:
    // Shared constant physics properties
    const RayleighNativeRef& shared_;
    // Incident gamma energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Id of element
    ElementId element_id_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "RayleighInteractor.i.hh"
