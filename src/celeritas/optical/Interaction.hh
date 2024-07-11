//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Interaction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "geocel/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * The result of a discrete optical interaction.
 *
 * All optical interactions are discrete. The wavelength of a photon is only
 * changed through absorption re-emission processes.
 */
struct Interaction
{
    //! Interaction result category
    enum class Action
    {
        scattered,  //!< Still alive, state has changed
        absorbed,  //!< Absorbed by the material
        unchanged,  //!< No state change, no secondaries
    };

    Real3 direction;  //!< Post-interaction direction
    Real3 polarization;  //!< Post-interaction polarization
    Action action{Action::scattered};  //!< Flags for interaction result

    //! Return an interaction respresenting an absorbed process
    static inline CELER_FUNCTION Interaction from_absorption();

    //! Return an interaction with no change in the track state
    static inline CELER_FUNCTION Interaction from_unchanged();

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
 * Construct an interaction for an absorbed optical photon.
 */
CELER_FUNCTION Interaction Interaction::from_absorption()
{
    Interaction result;
    result.action = Action::absorbed;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an interaction for edge cases where this is no state change.
 */
CELER_FUNCTION Interaction Interaction::from_unchanged()
{
    Interaction result;
    result.action = Action::unchanged;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
