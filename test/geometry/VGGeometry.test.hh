//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometry.test.hh
//---------------------------------------------------------------------------//

#include <vector>
#include "geometry/VGStateView.hh"
#include "geometry/VGView.hh"

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
struct VGGTestInit
{
    Real3 pos;
    Real3 dir;
};

//! Input data
struct VGGTestInput
{
    std::vector<VGGTestInit> init;
    int                      max_segments = 0;
    VGView                   shared;
    VGStateView              state;
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

//---------------------------------------------------------------------------//
} // namespace celeritas_test
