//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "LDemoInterface.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
void pre_step(const ParamsDeviceRef&, const StateDeviceRef&);

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_CUDA
inline void pre_step(const ParamsDeviceRef&, const StateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif
//---------------------------------------------------------------------------//
} // namespace demo_loop
