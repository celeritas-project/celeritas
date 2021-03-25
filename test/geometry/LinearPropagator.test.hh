//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagator.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/Assert.hh"
#include "geometry/GeoInterface.hh"

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
using LinPropTestInit = celeritas::GeoTrackInitializer;

//! Input data
struct LinPropTestInput
{
    std::vector<LinPropTestInit> init;
    int                          max_segments = 0;
    GeoParamsCRefDevice          params;
    GeoStateRefDevice            state;
};

//---------------------------------------------------------------------------//
//! Output results
struct LinPropTestOutput
{
    std::vector<int>    ids;
    std::vector<double> distances;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
LinPropTestOutput linprop_test(LinPropTestInput);

#if !CELERITAS_USE_CUDA
LinPropTestOutput linprop_test(LinPropTestInput)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
