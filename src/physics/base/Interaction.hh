//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Interaction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/ArrayUtils.hh"
#include "base/SoftEqual.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "sim/Action.hh"
#include "Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Change in state due to an interaction.
 */
struct Interaction
{
    Action           action;            //!< Failure, scatter, absorption, ...
    units::MevEnergy energy;            //!< Post-interaction energy
    Real3            direction;         //!< Post-interaction direction
    Secondary        secondary;         //!< First secondary is preallocated
    Span<Secondary>  secondaries;       //!< Remaining emitted secondaries
    units::MevEnergy energy_deposition; //!< Energy loss locally to material

    // Return an interaction representing a recoverable error
    static inline CELER_FUNCTION Interaction from_failure();

    // Return an interaction representing an absorbed process
    static inline CELER_FUNCTION Interaction from_absorption();

    // Return an interaction with no change in the particle's state
    static inline CELER_FUNCTION Interaction
    from_unchanged(units::MevEnergy energy, const Real3& direction);

    // Return an interaction indicating all state changes have been applied
    static inline CELER_FUNCTION Interaction from_processed();

    // Return an interaction representing the creation of a new track
    static inline CELER_FUNCTION Interaction from_spawned();

    // Whether the interaction succeeded
    explicit inline CELER_FUNCTION operator bool() const;

    // Total number of secondaries
    inline CELER_FUNCTION size_type num_secondaries() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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

//---------------------------------------------------------------------------//
/*!
 * Construct an interaction from a particle that was totally absorbed.
 */
CELER_FUNCTION Interaction Interaction::from_absorption()
{
    Interaction result;
    result.action    = Action::absorbed;
    result.energy    = zero_quantity();
    result.direction = {0, 0, 0};
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an interaction for edge cases where there is no state change.
 */
CELER_FUNCTION Interaction Interaction::from_unchanged(units::MevEnergy energy,
                                                       const Real3& direction)
{
    CELER_EXPECT(energy > zero_quantity());
    CELER_EXPECT(is_soft_unit_vector(direction));

    Interaction result;
    result.action    = Action::unchanged;
    result.energy    = energy;
    result.direction = direction;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct for end of step when all interaction data has been processed.
 */
CELER_FUNCTION Interaction Interaction::from_processed()
{
    Interaction result;
    result.action = Action::processed;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an interaction for the creation of a new track.
 */
CELER_FUNCTION Interaction Interaction::from_spawned()
{
    Interaction result;
    result.action = Action::spawned;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the interaction succeeded.
 */
CELER_FUNCTION Interaction::operator bool() const
{
    return action_completed(this->action);
}

//---------------------------------------------------------------------------//
/*!
 * Total number of secondaries.
 */
CELER_FUNCTION size_type Interaction::num_secondaries() const
{
    return (secondary ? 1 : 0) + secondaries.size();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
