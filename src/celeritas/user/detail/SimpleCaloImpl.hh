//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/SimpleCaloImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Types.hh"

#include "../SimpleCaloData.hh"
#include "../StepData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
void simple_calo_accum(HostRef<StepStateData> const& step,
                       HostRef<SimpleCaloStateData>& calo);

void simple_calo_accum(DeviceRef<StepStateData> const& step,
                       DeviceRef<SimpleCaloStateData>& calo);

#if !CELERITAS_USE_DEVICE
inline void simple_calo_accum(DeviceRef<StepStateData> const&,
                              DeviceRef<SimpleCaloStateData>&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
