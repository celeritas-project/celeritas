//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Tracking.hh
//---------------------------------------------------------------------------//
#pragma once

#include <limits>

#include "corecel/Types.hh"
namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Hard cutoffs for counters.
 */
struct TrackingLimits
{
    //! Don't limit the number of steps
    static inline constexpr size_type unlimited
        = std::numeric_limits<size_type>::max();

    //! Steps per track before killing it
    size_type steps{unlimited};
    //! Step iterations before aborting a run
    size_type step_iters{unlimited};
    //! Step iterations before aborting the optical stepping loop
    size_type optical_step_iters{unlimited};
    //! Integration substeps during field propagation before ending the step
    size_type field_substeps{100};

    //! Stop electron/positron below this energy
    // TODO: Energy electron_energy = Energy{0.001};
};

//---------------------------------------------------------------------------//
/*!
 * Specify non-physical parameters which can affect the physics.
 */
struct Tracking
{
    //! Hard-coded cutoffs before giving up
    TrackingLimits limits;

    //! Hardcoded maximum step for debugging charged particles (none if zero)
    real_type force_step_limit{};
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
