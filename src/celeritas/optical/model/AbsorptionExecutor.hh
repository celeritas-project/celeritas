//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/AbsorptionExecutor.hh
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
struct AbsorptionExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const&);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION void AbsorptionExecutor::operator()(CoreTrackView const&) {}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
