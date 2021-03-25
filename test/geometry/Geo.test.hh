//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTrackView.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "geometry/GeoInterface.hh"
#include "base/Assert.hh"

namespace celeritas_test
{
using celeritas::MemSpace;
using celeritas::Ownership;

using GeoParamsCRefDevice
    = celeritas::GeoParamsData<Ownership::const_reference, MemSpace::device>;
using GeoStateRefDevice
    = celeritas::GeoStateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
using VGGTestInit = celeritas::GeoTrackInitializer;

//! Input data
struct VGGTestInput
{
    std::vector<VGGTestInit> init;
    int                      max_segments = 0;
    GeoParamsCRefDevice      params;
    GeoStateRefDevice        state;
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
} // namespace celeritas_test
