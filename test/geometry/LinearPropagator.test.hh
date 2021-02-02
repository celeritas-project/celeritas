//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagator.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "geometry/GeoInterface.hh"
#include "geometry/LinearPropagator.hh"

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
using LinPropTestInit = GeoStateInitializer;

//! Input data
struct LinPropTestInput
{
    std::vector<LinPropTestInit> init;
    int                          max_segments = 0;
    GeoParamsPointers            shared;
    GeoStatePointers             state;
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
LinPropTestOutput linProp_test(LinPropTestInput);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
