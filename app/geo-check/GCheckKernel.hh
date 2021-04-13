//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GCheckKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geometry/GeoInterface.hh"

namespace geo_check
{
//---------------------------------------------------------------------------//
//! Run tracking on the CPU
void run_cpu(celeritas::GeoParamsPointers const&   params_view,
             celeritas::GeoStatePointers const&    track,
             celeritas::GeoStateInitializer const* init,
             int                                   max_steps);

//---------------------------------------------------------------------------//
//! Run tracking on the GPU
void run_gpu(const celeritas::GeoParamsPointers&   geo_params,
             const celeritas::GeoStatePointers&    geo_state,
             const celeritas::GeoStateInitializer& track_init,
             int                                   max_steps);

//---------------------------------------------------------------------------//
} // namespace geo_check
