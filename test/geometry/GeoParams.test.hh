//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParams.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "geometry/GeoStatePointers.hh"
#include "geometry/GeoParamsPointers.hh"
#include "geometry/GeoTrackView.hh"

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

//! Input data
struct GPTestInput
{
    int               max_segments = 0;
    GeoParamsPointers shared;
};

//---------------------------------------------------------------------------//
//! Output results
struct GPTestOutput
{
    std::vector<Real3>  pos;
    std::vector<double> vols;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
GPTestOutput gp_test(GPTestInput);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
