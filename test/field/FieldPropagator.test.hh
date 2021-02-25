//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagator.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include "FieldTestParams.hh"

#include <vector>
#include "geometry/GeoInterface.hh"
#include "geometry/GeoParams.hh"
#include "physics/base/ParticleInterface.hh"
#include "field/FieldParamsPointers.hh"

namespace celeritas_test
{
using namespace celeritas;

using ParticleParamsPointers
    = ParticleParamsData<Ownership::const_reference, MemSpace::device>;
using ParticleStatePointers
    = ParticleStateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

//! Input data
struct FPTestInput
{
    std::vector<GeoStateInitializer> init_geo;
    GeoParamsPointers                geo_params;
    GeoStatePointers                 geo_states;

    std::vector<ParticleTrackState> init_track;
    ParticleParamsPointers          particle_params;
    ParticleStatePointers           particle_states;

    FieldParamsPointers field_params;
    FieldTestParams     test_params;
};

//---------------------------------------------------------------------------//
//! Output results
struct FPTestOutput
{
    std::vector<double> step;
    std::vector<double> pos;
    std::vector<double> mom;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
FPTestOutput fp_test(FPTestInput input);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
