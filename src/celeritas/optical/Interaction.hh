//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Interaction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "geocel/Types.hh"

#include "WavelengthShiftData.hh"

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
        failed,  //!< Ran out of memory during sampling
    };

    Real3 direction;  //!< Post-interaction direction
    Real3 polarization;  //!< Post-interaction polarization
    Action action{Action::scattered};  //!< Flags for interaction result
    WlsDistributionData distribution;  //!< Data for generating WLS secondaries

    //! Return an interaction respresenting an absorbed process
    static inline CELER_FUNCTION Interaction from_absorption();

    //! Return an interaction with no change in the track state
    static inline CELER_FUNCTION Interaction from_unchanged();

    // Return an interaction representing a recoverable error
    static inline CELER_FUNCTION Interaction from_failure();

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
/*!
 * Indicate a failure to allocate memory for secondaries.
 */
CELER_FUNCTION Interaction Interaction::from_failure()
{
    Interaction result;
    result.action = Action::failed;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
