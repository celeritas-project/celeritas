//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Interaction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/ArrayUtils.hh"
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
    Span<Secondary>  secondaries;       //!< Emitted secondaries
    units::MevEnergy energy_deposition; //!< Energy loss locally to material

    // Return an interaction representing a recoverable error
    static inline CELER_FUNCTION Interaction from_failure();

    // Return an interaction representing an absorbed process
    static inline CELER_FUNCTION Interaction from_absorption();

    // Return an interaction with no change in the particle's state
    static inline CELER_FUNCTION Interaction
    from_unchanged(units::MevEnergy energy, const Real3& direction);

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
    CELER_EXPECT(energy.value() > 0);
    CELER_EXPECT(is_soft_unit_vector(direction, SoftEqual<real_type>()));

    Interaction result;
    result.action    = Action::unchanged;
    result.energy    = energy;
    result.direction = direction;
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
} // namespace celeritas
