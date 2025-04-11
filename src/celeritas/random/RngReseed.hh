//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngReseed.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/random/data/RngData.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Reinitialize the RNG states on host/device at the start of an event
void reseed_rng(DeviceCRef<RngParamsData> const&,
                DeviceRef<RngStateData> const&,
                StreamId,
                UniqueEventId);

void reseed_rng(HostCRef<RngParamsData> const&,
                HostRef<RngStateData> const&,
                StreamId,
                UniqueEventId);

#if !CELER_USE_DEVICE && (!defined(__DOXYGEN__) || __DOXYGEN__ > 0x010908)
//---------------------------------------------------------------------------//
/*!
 * Reinitialize the RNG states on device at the start of an event.
 */
inline void reseed_rng(DeviceCRef<RngParamsData> const&,
                       DeviceRef<RngStateData> const&,
                       StreamId,
                       UniqueEventId)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
