//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DiscreteSelectExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../CoreTrackView.hh"
#include "../PhysicsData.hh"
#include "../PhysicsStepUtils.hh"
#include "../PhysicsTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
struct DiscreteSelectExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION void DiscreteSelectExecutor::operator()(CoreTrackView const&)
{
    // TODO: Implement executor
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
