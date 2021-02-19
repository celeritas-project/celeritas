//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoKernel.cu
//---------------------------------------------------------------------------//
#include "KNDemoKernel.hh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "base/ArrayUtils.hh"
#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/em/detail/KleinNishinaInteractor.hh"
#include "random/cuda/RngEngine.hh"
#include "physics/grid/PhysicsGridCalculator.hh"
#include "DetectorView.hh"
#include "KernelUtils.hh"

using namespace celeritas;
using celeritas::detail::KleinNishinaInteractor;

namespace demo_interactor
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Kernel to initialize particle data.
 *
 * For testing purposes (this might not be the case for the final app) we use a
 * grid-stride loop rather than requiring that each thread correspond exactly
 * to a particle track. In other words, this method allows a single warp to
 * operate on two 32-thread chunks of data.
 *  https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 */
__global__ void initialize_kernel(ParamsDeviceRef const params,
                                  StateDeviceRef const  states,
                                  InitialPointers const init)
{
    // Grid-stride loop, see
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < static_cast<int>(states.size());
         tid += blockDim.x * gridDim.x)
    {
        ParticleTrackView particle(
            params.particle, states.particle, ThreadId(tid));
        particle = init.particle;

        // Particles begin alive and in the +z direction
        states.direction[tid] = {0, 0, 1};
        states.position[tid]  = {0, 0, 0};
        states.time[tid]      = 0;
        states.alive[tid]     = true;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Perform a single interaction per particle track.
 *
 * The interaction:
 * - Clears the energy deposition
 * - Samples the KN interaction
 * - Allocates and emits a secondary
 * - Kills the secondary, depositing its local energy
 * - Applies the interaction (updating track direction and energy)
 */
__global__ void iterate_kernel(ParamsDeviceRef const            params,
                               StateDeviceRef const             states,
                               SecondaryAllocatorPointers const secondaries,
                               DetectorPointers const           detector)
{
    SecondaryAllocatorView allocate_secondaries(secondaries);
    DetectorView           detector_hit(detector);
    PhysicsGridCalculator  calc_xs(params.tables.xs, params.tables.reals);

    for (unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < states.size();
         tid += blockDim.x * gridDim.x)
    {
        // Skip loop if already dead
        if (!states.alive[tid])
        {
            continue;
        }

        // Construct particle accessor from immutable and thread-local data
        ParticleTrackView particle(
            params.particle, states.particle, ThreadId(tid));
        RngEngine rng(states.rng, ThreadId(tid));

        // Move to collision
        demo_interactor::move_to_collision(particle,
                                           calc_xs,
                                           states.direction[tid],
                                           &states.position[tid],
                                           &states.time[tid],
                                           rng);

        Hit h;
        h.pos    = states.position[tid];
        h.thread = ThreadId(tid);
        h.time   = states.time[tid];

        if (particle.energy() < units::MevEnergy{0.01})
        {
            // Particle is below interaction energy
            h.dir              = states.direction[tid];
            h.energy_deposited = particle.energy();

            // Deposit energy and kill
            detector_hit(h);
            states.alive[tid] = false;
            continue;
        }

        // Construct RNG and interaction interfaces
        KleinNishinaInteractor interact(params.kn_interactor,
                                        particle,
                                        states.direction[tid],
                                        allocate_secondaries);

        // Perform interaction: should emit a single particle (an electron)
        Interaction interaction = interact(rng);
        CELER_ASSERT(interaction);
        CELER_ASSERT(interaction.secondaries.size() == 1);

        // Deposit energy from the secondary (effectively, an infinite energy
        // cutoff)
        {
            const auto& secondary = interaction.secondaries.front();
            h.dir                 = secondary.direction;
            h.energy_deposited    = secondary.energy;
            detector_hit(h);
        }

        // Update post-interaction state (apply interaction)
        states.direction[tid] = interaction.direction;
        particle.energy(interaction.energy);
    }
}
} // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
/*!
 * Initialize particle states.
 */
void initialize(const CudaGridParams&  grid,
                const ParamsDeviceRef& params,
                const StateDeviceRef&  states,
                const InitialPointers& initial)
{
    // TODO: remove grid params in favor of one thread per track. In the
    // meantime, `calc_kernel_params` registers the kernel call and occupancy
    // with the diagnostics.
    static const KernelParamCalculator calc_kernel_params(
        initialize_kernel, "initialize", grid.block_size);
    calc_kernel_params(states.size());

    CELER_EXPECT(states.alive.size() == states.size());
    CELER_EXPECT(states.rng.size() == states.size());
    initialize_kernel<<<grid.grid_size, grid.block_size>>>(
        params, states, initial);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Run an iteration.
 */
void iterate(const CudaGridParams&              grid,
             const ParamsDeviceRef&             params,
             const StateDeviceRef&              states,
             const SecondaryAllocatorPointers&  secondaries,
             const celeritas::DetectorPointers& detector)
{
    // TODO: remove grid params, see above
    static const KernelParamCalculator calc_kernel_params(
        iterate_kernel, "iterate", grid.block_size);
    calc_kernel_params(states.size());

    iterate_kernel<<<grid.grid_size, grid.block_size>>>(
        params, states, secondaries, detector);
    CELER_CUDA_CHECK_ERROR();

    if (grid.sync)
    {
        // Note: the device synchronize is useful for debugging and necessary
        // for timing diagnostics.
        CELER_CUDA_CALL(cudaDeviceSynchronize());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sum the total number of living particles.
 */
size_type reduce_alive(Span<bool> alive, const CudaGridParams& grid)
{
    size_type result = thrust::reduce(
        thrust::device_pointer_cast(alive.data()),
        thrust::device_pointer_cast(alive.data() + alive.size()),
        size_type(0),
        thrust::plus<size_type>());
    CELER_CUDA_CHECK_ERROR();

    if (grid.sync)
    {
        CELER_CUDA_CALL(cudaDeviceSynchronize());
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
