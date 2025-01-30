//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../CoreTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
struct RayleighExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const&);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION void RayleighExecutor::operator()(CoreTrackView const&) {}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
