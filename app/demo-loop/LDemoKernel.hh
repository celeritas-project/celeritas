//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "sim/TrackData.hh"

using celeritas::ParamsDeviceRef;
using celeritas::ParamsHostRef;
using celeritas::StateDeviceRef;
using celeritas::StateHostRef;

namespace demo_loop
{
//---------------------------------------------------------------------------//
// TODO: convert all these to functors and autogenerate kernels
//---------------------------------------------------------------------------//
void pre_step(const ParamsDeviceRef&, const StateDeviceRef&);
void pre_step(const ParamsHostRef&, const StateHostRef&);
void along_and_post_step(const ParamsDeviceRef&, const StateDeviceRef&);
void along_and_post_step(const ParamsHostRef&, const StateHostRef&);
void process_interactions(const ParamsDeviceRef&, const StateDeviceRef&);
void process_interactions(const ParamsHostRef&, const StateHostRef&);
void cleanup(const ParamsDeviceRef&, const StateDeviceRef&);
void cleanup(const ParamsHostRef&, const StateHostRef&);

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_CUDA
inline void pre_step(const ParamsDeviceRef&, const StateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}

inline void along_and_post_step(const ParamsDeviceRef&, const StateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}

inline void process_interactions(const ParamsDeviceRef&, const StateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}

inline void cleanup(const ParamsDeviceRef&, const StateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif
//---------------------------------------------------------------------------//
} // namespace demo_loop
