//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalInteraction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "geocel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 */
struct OpticalInteraction
{
    //! Interaction result category
    enum class Action
    {
        scattered,  //!< Still alive, state has changed
        absorbed,  //!< Absorbed by the material
        unchanged,  //!< No state change, no secondaries
        failed,  //!< Ran out of memory during sampling
    };

    Real3 direction;  //!< Post-interaction direction
    Real3 polarization;  //!< Post-interaction polarization
    Action action{Action::scattered};  //!< Flags for interaction result

    //! Return an interaction representing a recoverable error
    static inline CELER_FUNCTION OpticalInteraction from_failure();

    //! Return an interaction respresenting an absorbed process
    static inline CELER_FUNCTION OpticalInteraction from_absorption();

    //! Return an interaction with no change in the track state
    static inline CELER_FUNCTION OpticalInteraction from_unchanged();

    //! Whether the state changed but did not fail
    CELER_FUNCTION bool changed() const
    {
        return static_cast<int>(action) < static_cast<int>(Action::unchanged);
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Indicate a failure to allocate memory for secondaries.
 */
CELER_FUNCTION OpticalInteraction OpticalInteraction::from_failure()
{
    OpticalInteraction result;
    result.action = Action::failed;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an interaction for an absorbed optical photon.
 */
CELER_FUNCTION OpticalInteraction OpticalInteraction::from_absorption()
{
    OpticalInteraction result;
    result.action = Action::absorbed;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an interaction for edge cases where this is no state change.
 */
CELER_FUNCTION OpticalInteraction OpticalInteraction::from_unchanged()
{
    OpticalInteraction result;
    result.action = Action::unchanged;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
