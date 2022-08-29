//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/Vecgeom.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "celeritas/ext/VecgeomData.hh"

namespace celeritas
{
namespace test
{

using GeoParamsCRefDevice = DeviceCRef<celeritas::VecgeomParamsData>;
using GeoStateRefDevice   = DeviceRef<celeritas::VecgeomStateData>;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

//! Input data
struct VGGTestInput
{
    std::vector<GeoTrackInitializer> init;
    int                              max_segments = 0;
    GeoParamsCRefDevice              params;
    GeoStateRefDevice                state;
};

//---------------------------------------------------------------------------//
//! Output results
struct VGGTestOutput
{
    std::vector<int>    ids;
    std::vector<double> distances;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
VGGTestOutput vgg_test(VGGTestInput);

#if !CELERITAS_USE_CUDA
inline VGGTestOutput vgg_test(VGGTestInput)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
