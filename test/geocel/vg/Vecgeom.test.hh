//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/Vecgeom.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "geocel/vg/VecgeomData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

//! Input data
struct VGGTestInput
{
    std::vector<GeoTrackInitializer> init;
    int max_segments = 0;
    DeviceCRef<VecgeomParamsData> params;
    DeviceRef<VecgeomStateData> state;
};

//---------------------------------------------------------------------------//
//! Output results
struct VGGTestOutput
{
    std::vector<int> ids;
    std::vector<double> distances;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
VGGTestOutput vgg_test(VGGTestInput const&);

#if !CELERITAS_USE_CUDA
inline VGGTestOutput vgg_test(VGGTestInput const&)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
