//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/G4VecgeomConversion.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "celeritas/ext/VecgeomData.hh"

namespace celeritas_test
{
using celeritas::MemSpace;
using celeritas::Ownership;

using GeoParamsCRefDevice
    = celeritas::VecgeomParamsData<Ownership::const_reference, MemSpace::device>;
using GeoStateRefDevice
    = celeritas::VecgeomStateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

//! Input data
struct G4VGConvTestInput
{
    using GeoTrackInitializer = celeritas::GeoTrackInitializer;

    std::vector<GeoTrackInitializer> init;
    int                              max_segments = 0;
    GeoParamsCRefDevice              params;
    GeoStateRefDevice                state;
};

//---------------------------------------------------------------------------//
//! Output results
struct G4VGConvTestOutput
{
    std::vector<int>    ids;
    std::vector<double> distances;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
G4VGConvTestOutput g4vgconv_test(G4VGConvTestInput);

#if !CELERITAS_USE_CUDA
inline G4VGConvTestOutput g4vgconv_test(G4VGConvTestInput)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
