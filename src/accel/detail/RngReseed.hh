//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/RngReseed.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/random/RngData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Reinitialize the RNG states on host/device using the Geant4 Event ID
void reseed_rng(DeviceCRef<RngParamsData> const&,
                DeviceRef<RngStateData> const&,
                size_type);

void reseed_rng(HostCRef<RngParamsData> const&,
                HostRef<RngStateData> const&,
                size_type);

#if !CELER_USE_DEVICE
//---------------------------------------------------------------------------//
/*!
 * Reinitialize the RNG states on device using the Geant4 Event ID.
 */
inline void reseed_rng(DeviceCRef<RngParamsData> const&,
                       DeviceRef<RngStateData> const&,
                       size_type)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
