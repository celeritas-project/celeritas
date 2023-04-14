//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromSecondariesAction.cc
//---------------------------------------------------------------------------//
#include "ExtendFromSecondariesAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/track/TrackInitUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data
 */
void ExtendFromSecondariesAction::execute(ParamsHostCRef const& params,
                                          StateHostRef& states) const
{
    extend_from_secondaries(params, states);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data
 */
void ExtendFromSecondariesAction::execute(ParamsDeviceCRef const& params,
                                          StateDeviceRef& states) const
{
    extend_from_secondaries(params, states);
}

}  // namespace celeritas
