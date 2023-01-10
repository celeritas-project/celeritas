//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GCheckKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "orange/Types.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoParamsFwd.hh"
#include "celeritas/geo/GeoTrackView.hh"

namespace geo_check
{
//---------------------------------------------------------------------------//
using celeritas::GeoTrackInitializer;

using GeoParamsCRefHost   = celeritas::HostCRef<celeritas::GeoParamsData>;
using GeoParamsCRefDevice = celeritas::DeviceCRef<celeritas::GeoParamsData>;
using GeoStateRefDevice   = celeritas::DeviceRef<celeritas::GeoStateData>;

using SPConstGeo = std::shared_ptr<const celeritas::GeoParams>;

//---------------------------------------------------------------------------//
//! Input and return structs
struct GCheckInput
{
    std::vector<celeritas::GeoTrackInitializer> init;
    int                                         max_steps = 0;
    GeoParamsCRefDevice                         params;
    GeoStateRefDevice                           state;
};

//! Output results
struct GCheckOutput
{
    std::vector<int>    ids;
    std::vector<double> distances;
};

//---------------------------------------------------------------------------//
CELER_FORCEINLINE_FUNCTION int physid(const celeritas::GeoTrackView& geo)
{
    if (geo.is_outside())
        return 0;
#if CELERITAS_USE_VECGEOM
    return geo.volume_physid();
#else
    return geo.volume_id().get();
#endif
}

//---------------------------------------------------------------------------//
//! Run tracking on the CPU
GCheckOutput run_cpu(const SPConstGeo&          geo_params,
                     const GeoTrackInitializer* track_init,
                     int                        max_steps);

//! Run tracking on the GPU
GCheckOutput run_gpu(GCheckInput init);

#if !CELERITAS_USE_CUDA
inline GCheckOutput run_gpu(GCheckInput)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace geo_check
