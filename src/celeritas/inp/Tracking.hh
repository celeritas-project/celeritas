//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Tracking.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Hard cutoffs for counter.
 */
struct TrackingLimits
{
    //! Don't limit the number of steps
    static inline constexpr size_type unlimited
        = numeric_limits<size_type>::max();

    //! Steps per track before killing it
    size_type steps{unlimited};
    //! Step iterations before aborting a run
    size_type step_iters{unlimited};
    //! Integration substeps during field propagation before ending the step
    size_type field_substeps{100};
};

//---------------------------------------------------------------------------//
/*!
 * Specify non-physical parameters which can affect the physics.
 */
struct Tracking
{
    //! Hard-coded cutoffs before giving up
    TrackingLimits limits;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
