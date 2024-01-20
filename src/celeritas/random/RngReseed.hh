//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngReseed.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"

#include "RngData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Reinitialize the RNG states on host/device at the start of an event
void reseed_rng(DeviceCRef<RngParamsData> const&,
                DeviceRef<RngStateData> const&,
                size_type);

void reseed_rng(HostCRef<RngParamsData> const&,
                HostRef<RngStateData> const&,
                size_type);

#if !CELER_USE_DEVICE
//---------------------------------------------------------------------------//
/*!
 * Reinitialize the RNG states on device at the start of an event.
 */
inline void reseed_rng(DeviceCRef<RngParamsData> const&,
                       DeviceRef<RngStateData> const&,
                       size_type)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
