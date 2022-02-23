//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyDiagnostic.cu
//---------------------------------------------------------------------------//
#include "EnergyDiagnostic.hh"

#include "base/CollectionBuilder.hh"
#include "base/KernelParamCalculator.device.hh"
#include "base/Macros.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Get energy deposition from state data and accumulate in appropriate bin
 */
__global__ void
bin_energy_kernel(const StateDeviceRef states, PointersDevice pointers)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    EnergyDiagnosticLauncher<MemSpace::device> launch(states, pointers);
    launch(tid);
}

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
void bin_energy(const StateDeviceRef& states, PointersDevice& pointers)
{
    static const KernelParamCalculator calc_launch_params(bin_energy_kernel,
                                                          "bin_energy");
    auto lparams = calc_launch_params(states.size());
    bin_energy_kernel<<<lparams.grid_size, lparams.block_size>>>(states,
                                                                 pointers);
    CELER_DEVICE_CHECK_ERROR();
}

} // namespace demo_loop
