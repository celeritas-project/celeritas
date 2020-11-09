//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Interaction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "sim/Action.hh"
#include "Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Change in state to a particle during an interaction.
 *
 * TODO: we might also need local enery deposition for some inelastic
 * processes/models. (?)
 */
struct Interaction
{
    Action           action;      //!< Failure, scatter, absorption, ...
    units::MevEnergy energy;      //!< Post-interaction energy
    Real3            direction;   //!< Post-interaction direction
    span<Secondary>  secondaries; //!< Emitted secondaries

    // Return an interaction representing a recoverable error
    static inline CELER_FUNCTION Interaction from_failure();

    // Return an interaction representing an absorbed process
    static inline CELER_FUNCTION Interaction from_absorption();

    // Whether the interaction succeeded
    explicit inline CELER_FUNCTION operator bool() const;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION Interaction Interaction::from_failure()
{
    Interaction result;
    result.action = Action::failed;
    return result;
}

/*!
 * An interaction with an absorbed process.
 */
CELER_FUNCTION Interaction Interaction::from_absorption()
{
    Interaction result;
    result.action    = Action::absorbed;
    result.energy    = units::MevEnergy{0};
    result.direction = {0, 0, 0};
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the interaction succeeded
 */
CELER_FUNCTION Interaction::operator bool() const
{
    return action_completed(this->action);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
