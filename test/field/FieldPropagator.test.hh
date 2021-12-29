//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagator.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "field/FieldParamsData.hh"
#include "geometry/GeoData.hh"
#include "geometry/Types.hh"
#include "physics/base/ParticleData.hh"
#include "FieldTestParams.hh"

namespace celeritas_test
{
using celeritas::MemSpace;
using celeritas::Ownership;

using GeoParamsCRefDevice
    = celeritas::GeoParamsData<Ownership::const_reference, MemSpace::device>;
using GeoStateRefDevice
    = celeritas::GeoStateData<Ownership::reference, MemSpace::device>;

using ParticleParamsRef
    = celeritas::ParticleParamsData<Ownership::const_reference, MemSpace::device>;
using ParticleStateRef
    = celeritas::ParticleStateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct FPTestInput
{
    using GeoInit      = celeritas::GeoTrackInitializer;
    using ParticleInit = celeritas::ParticleTrackInitializer;

    std::vector<GeoInit> init_geo;
    GeoParamsCRefDevice  geo_params;
    GeoStateRefDevice    geo_states;

    std::vector<ParticleInit> init_track;
    ParticleParamsRef         particle_params;
    ParticleStateRef          particle_states;

    celeritas::FieldParamsData field_params;
    FieldTestParams            test;
};

//---------------------------------------------------------------------------//
//! Output results
struct FPTestOutput
{
    std::vector<double> step;
    std::vector<double> pos;
    std::vector<double> dir;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
FPTestOutput fp_test(FPTestInput input);
FPTestOutput bc_test(FPTestInput input);

#if !CELERITAS_USE_CUDA
inline FPTestOutput fp_test(FPTestInput)
{
    CELER_NOT_CONFIGURED("CUDA");
}
inline FPTestOutput bc_test(FPTestInput)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif
//---------------------------------------------------------------------------//
} // namespace celeritas_test
