//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Interaction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "Secondary.hh"
#if CELERITAS_DEBUG
#    include "corecel/math/NumericLimits.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Change in state due to an interaction.
 */
struct Interaction
{
    using Energy = units::MevEnergy;

    //! Interaction result category
    enum class Action
    {
        scattered,  //!< Still alive, state has changed
        absorbed,  //!< Absorbed or transformed to another particle type
        onloaded,  //!< Transfer interaction to the host
        unchanged,  //!< No state change, no secondaries
        failed,  //!< Ran out of memory during sampling
    };

    Energy energy;  //!< Post-interaction energy
    Real3 direction;  //!< Post-interaction direction
    Span<Secondary> secondaries;  //!< Emitted secondaries
    Energy energy_deposition{0};  //!< Energy loss locally to material
    Action action{Action::scattered};  //!< Flags for interaction result

    // Return an interaction representing a recoverable error
    static inline CELER_FUNCTION Interaction from_failure();

    // Return an interaction representing an absorbed process
    static inline CELER_FUNCTION Interaction from_absorption();

    // Return an interaction with no change in the track state
    static inline CELER_FUNCTION Interaction from_unchanged();

    // Return an interaction with onload in the tract state
    static inline CELER_FUNCTION Interaction from_onloaded();

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
 * \todo Document and/or refactor into a class that hides details
 * - alpha == small_step_alpha() ? "true path is very small" (true path scaling
 *   changes)
 * - is_displaced == false ? limit_min is unchanged and alpha ==
 *   small_step_alpha()
 * - true_step >= geom_path
 *
 * The value \f$ \alpha \f$ is used in the approximation of the MSC
 * transport cross section as a linear function over the current step. It is
 * the negative slope of the transport MFP from start to stop, divided by the
 * starting MFP. (Since the transport cross section generally decreases
 * monotonically with increasing energy over the step, alpha will usually be
 * positive or zero for the step. Some known errors in the cross sections for
 * positrons result in negative alpha around a discontinuity at 10 MeV.)
 */
struct MscStep
{
    //! Use a small step approximation for the path length correction
    static CELER_CONSTEXPR_FUNCTION real_type small_step_alpha() { return 0; }

    bool is_displaced{true};  //!< Flag for the lateral displacement
    real_type true_path{};  //!< True path length due to the msc [len]
    real_type geom_path{};  //!< Geometrical path length [len]
    real_type alpha = small_step_alpha();  //!< Scaled MFP slope [1/len]
};

//---------------------------------------------------------------------------//
/*!
 * Persistent range properties for multiple scattering (msc) within a volume.
 *
 * These values are calculated at the first step in every msc tracking volume
 * and reused at subsequent steps within the same volume.
 *
 * \todo move to physics step data
 */
struct MscRange
{
    real_type range_init{};  //!< Initial msc range [len]
    real_type range_factor{};  //!< Scale factor for the msc range
    real_type limit_min{};  //!< Minimum of the true path limit [len]

    explicit CELER_FUNCTION operator bool() const
    {
        return range_init > 0 && range_factor > 0 && limit_min > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Result of multiple scattering.
 */
struct MscInteraction
{
    //! Interaction result category
    enum class Action
    {
        unchanged,  //!< No state change
        scattered,  //!< Only direction changed
        displaced,  //!< Direction and position changed
        size_
    };

    Real3 direction;  //!< Post-step direction
    Real3 displacement;  //!< Lateral displacement
    Action action{Action::unchanged};  //!< Flags for interaction result
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
#if CELERITAS_DEBUG
    // Direction should *not* be accessed if incident particle is absorbed.
    constexpr auto nan = numeric_limits<real_type>::quiet_NaN();
    result.direction = {nan, nan, nan};
#endif
    result.action = Action::absorbed;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an interaction for edge cases where there is no state change.
 */
CELER_FUNCTION Interaction Interaction::from_unchanged()
{
    Interaction result;
    result.action = Action::unchanged;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an interaction from a particle that will be onloaded.
 */
CELER_FUNCTION Interaction Interaction::from_onloaded()
{
    Interaction result;
    result.action = Action::onloaded;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
