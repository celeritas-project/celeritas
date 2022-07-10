//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldPropagator.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "orange/Types.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/phys/ParticleData.hh"

#include "FieldTestParams.hh"

namespace celeritas_test
{

using GeoParamsCRefDevice = celeritas::DeviceCRef<celeritas::GeoParamsData>;
using GeoStateRefDevice   = celeritas::DeviceRef<celeritas::GeoStateData>;
using ParticleParamsRef = celeritas::DeviceCRef<celeritas::ParticleParamsData>;
using ParticleStateRef  = celeritas::DeviceRef<celeritas::ParticleStateData>;

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

    celeritas::FieldDriverOptions field_params;
    FieldTestParams               test;
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

#if !CELER_USE_DEVICE
inline FPTestOutput fp_test(FPTestInput)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
inline FPTestOutput bc_test(FPTestInput)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif
//---------------------------------------------------------------------------//
} // namespace celeritas_test
