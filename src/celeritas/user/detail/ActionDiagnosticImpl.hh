//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/ActionDiagnosticImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackData.hh"

#include "../ActionDiagnosticData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
void tally_action(HostCRef<CoreParamsData> const&,
                  HostRef<CoreStateData> const&,
                  HostRef<ActionDiagnosticStateData>&);

void tally_action(DeviceCRef<CoreParamsData> const&,
                  DeviceRef<CoreStateData> const&,
                  DeviceRef<ActionDiagnosticStateData>&);

#if !CELER_USE_DEVICE
inline void tally_action(DeviceCRef<CoreParamsData> const&,
                         DeviceRef<CoreStateData> const&,
                         DeviceRef<ActionDiagnosticStateData>&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
