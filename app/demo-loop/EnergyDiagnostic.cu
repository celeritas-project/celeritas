//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyDiagnostic.cu
//---------------------------------------------------------------------------//
#include "EnergyDiagnostic.hh"

#include "base/Atomics.hh"
#include "base/CollectionBuilder.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "base/Macros.hh"
#include "physics/grid/NonuniformGrid.hh"

namespace demo_loop
{
/*!
 * Accumulate energy deposition in diagnostic
 */
template<>
void EnergyDiagnostic<MemSpace::device>::end_step(const StateDataRef& states)
{
    // Set up pointers to pass to device
    EnergyBinPointers pointers;
    pointers.z_bounds    = z_bounds_;
    pointers.energy_by_z = energy_by_z_;

    // Invoke kernel for binning energies
    demo_loop::bin_energy(states, pointers);
}

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Get energy deposition from state data and accumulate in appropriate bin
 */
__global__ void
bin_energy_kernel(const StateDataRefDevice states, EnergyBinPointers pointers)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    // Create grid from EnergybinPointers
    NonuniformGrid<real_type> grid(pointers.z_bounds);

    real_type z_pos             = states.geometry.pos[tid][2];
    real_type energy_deposition = states.energy_deposition[tid];

    using BinId = ItemId<real_type>;
    if (z_pos > grid.front() && z_pos < grid.back())
    {
        auto bin = grid.find(z_pos);
        atomic_add(&pointers.energy_by_z[BinId{bin}], energy_deposition);
    }
}

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
void bin_energy(const StateDataRefDevice& states, EnergyBinPointers& pointers)
{
    static const celeritas::KernelParamCalculator calc_launch_params(
        bin_energy_kernel, "bin_energy");
    auto lparams = calc_launch_params(states.size());
    bin_energy_kernel<<<lparams.grid_size, lparams.block_size>>>(states,
                                                                 pointers);
    CELER_CUDA_CHECK_ERROR();
}

} // namespace demo_loop
