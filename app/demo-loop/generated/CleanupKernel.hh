//----------------------------------*-hh-*-----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CleanupKernel.hh
//! \note Auto-generated by gen-demo-loop-kernel.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "celeritas_config.h"
#include "base/Assert.hh"

namespace demo_loop
{
namespace generated
{
void cleanup(
    const celeritas::ParamsHostRef&,
    const celeritas::StateHostRef&);

void cleanup(
    const celeritas::ParamsDeviceRef&,
    const celeritas::StateDeviceRef&);

#if !CELERITAS_USE_CUDA
inline void cleanup(
    const celeritas::ParamsDeviceRef&,
    const celeritas::StateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

} // namespace generated
} // namespace demo_loop
