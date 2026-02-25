//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/FresnelReflectivityExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Return an interact result for the reflectivity action.
 */
struct FresnelReflectivityExecutor
{
    // Get trivial reflectivity action
    inline CELER_FUNCTION ReflectivityAction
    operator()(CoreTrackView const&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get trivial reflectivity action.
 *
 * Always returns a \c interact result for the reflectivity action.
 */
CELER_FUNCTION ReflectivityAction
FresnelReflectivityExecutor::operator()(CoreTrackView const&) const
{
    return ReflectivityAction::interact;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
