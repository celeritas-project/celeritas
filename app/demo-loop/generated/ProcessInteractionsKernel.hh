//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/generated/ProcessInteractionsKernel.hh
//! \note Auto-generated by gen-demo-loop-kernel.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "base/Assert.hh"
#include "base/Macros.hh"

namespace demo_loop
{
namespace generated
{
void process_interactions(
    const celeritas::ParamsHostRef&,
    const celeritas::StateHostRef&);

void process_interactions(
    const celeritas::ParamsDeviceRef&,
    const celeritas::StateDeviceRef&);

#if !CELER_USE_DEVICE
inline void process_interactions(
    const celeritas::ParamsDeviceRef&,
    const celeritas::StateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

} // namespace generated
} // namespace demo_loop
