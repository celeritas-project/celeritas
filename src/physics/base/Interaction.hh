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

#include "Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Change in state due to an interaction.
 */
struct Interaction
{
    //! Interaction result category
    enum class Action
    {
        scattered, //!< Still alive, state has changed
        absorbed,  //!< Absorbed or transformed to another particle type
        unchanged, //!< No state change, no secondaries
        failed,    //!< Ran out of memory during sampling
    };

    units::MevEnergy energy;               //!< Post-interaction energy
    Real3            direction;            //!< Post-interaction direction
    Span<Secondary>  secondaries;          //!< Emitted secondaries
    units::MevEnergy energy_deposition{0}; //!< Energy loss locally to material
    Action action{Action::scattered};      //!< Flags for interaction result

    // Return an interaction representing a recoverable error
    static inline CELER_FUNCTION Interaction from_failure();

    // Return an interaction representing an absorbed process
    static inline CELER_FUNCTION Interaction from_absorption();

    // Return an interaction with no change in the particle's state
    static inline CELER_FUNCTION Interaction
    from_unchanged(units::MevEnergy energy, const Real3& direction);

    //! Whether the state changed but did not fail
    CELER_FUNCTION bool changed() const
    {
        return static_cast<int>(action) < static_cast<int>(Action::unchanged);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Step lengths and properties needed to apply multiple scattering.
 *
 * \todo Document and/or refactor into a class that hides details:
 * - alpha < 0 ? "true path is very small" (true path scaling changes)
 * - is_displaced == false ? limit_min is unchanged and alpha < 0
 * - true_step >= geom_path
 */
struct MscStep
{
    bool      is_displaced{true}; //!< Flag for the lateral displacement
    real_type phys_step{};        //!< Step length from physics processes
    real_type true_path{};        //!< True path length due to the msc
    real_type geom_path{};        //!< Geometrical path length
    real_type limit_min{1e-8};    //!< Minimum of the true path limit
    real_type alpha{-1};          //!< An effecive mfp rate by distance
};

//---------------------------------------------------------------------------//
/*!
 * Result of multiple scattering.
 *
 * The "true" step length is the physical path length taken along the geometric
 * step, accounting for the extra distance taken between along-step
 * elastic collisions.
 */
struct MscInteraction
{
    real_type step_length;  //!< True step length
    Real3     direction;    //!< Post-step direction
    Real3     displacement; //!< Lateral displacement
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
/*!
 * Construct an interaction from a particle that was totally absorbed.
 */
CELER_FUNCTION Interaction Interaction::from_absorption()
{
    Interaction result;
    result.energy = zero_quantity();
    result.action = Action::absorbed;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an interaction for edge cases where there is no state change.
 */
CELER_FUNCTION Interaction Interaction::from_unchanged(units::MevEnergy,
                                                       const Real3&)
{
    Interaction result;
    result.action = Action::unchanged;
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
