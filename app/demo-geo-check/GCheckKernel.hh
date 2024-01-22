//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geo-check/GCheckKernel.hh
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
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/geo/GeoTrackView.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//

using GeoParamsCRefHost = HostCRef<GeoParamsData>;
using GeoParamsCRefDevice = DeviceCRef<GeoParamsData>;
using GeoStateRefDevice = DeviceRef<GeoStateData>;

using SPConstGeo = std::shared_ptr<GeoParams const>;

//---------------------------------------------------------------------------//
//! Input and return structs
struct GCheckInput
{
    std::vector<GeoTrackInitializer> init;
    int max_steps = 0;
    GeoParamsCRefDevice params;
    GeoStateRefDevice state;
};

//! Output results
struct GCheckOutput
{
    std::vector<int> ids;
    std::vector<double> distances;
};

//---------------------------------------------------------------------------//
CELER_FORCEINLINE_FUNCTION int physid(GeoTrackView const& geo)
{
    if (geo.is_outside())
        return 0;
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
    return geo.volume_physid();
#else
    return geo.volume_id().get();
#endif
}

//---------------------------------------------------------------------------//
//! Run tracking on the CPU
GCheckOutput run_cpu(SPConstGeo const& geo_params,
                     GeoTrackInitializer const* track_init,
                     int max_steps);

//! Run tracking on the GPU
GCheckOutput run_gpu(GCheckInput const& init);

#if !CELER_USE_DEVICE
inline GCheckOutput run_gpu(GCheckInput const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
