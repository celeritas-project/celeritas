//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/ParticleParamsPointers.hh"
#include "physics/base/ParticleStatePointers.hh"
#include "physics/base/SecondaryAllocatorPointers.hh"
#include "physics/em/KleinNishinaInteractorPointers.hh"

namespace celeritas
{
struct RngStatePointers;
}

namespace demo_interactor
{
//---------------------------------------------------------------------------//
//! Kernel thread dimensions
struct CudaGridParams
{
    unsigned int block_size = 256; //!< Threads per block
    unsigned int grid_size  = 32;  //!< Blocks per grid
};

//---------------------------------------------------------------------------//
// Initialize particle states
void initialize(CudaGridParams                    grid,
                celeritas::ParticleParamsPointers params,
                celeritas::ParticleStatePointers  states,
                celeritas::ParticleTrackState     initial_state,
                celeritas::RngStatePointers       rng_states,
                celeritas::span<celeritas::Real3> direction,
                celeritas::span<bool>             alive);

//---------------------------------------------------------------------------//
// Run an iteration
void iterate(CudaGridParams                            grid,
             celeritas::ParticleParamsPointers         params,
             celeritas::ParticleStatePointers          states,
             celeritas::KleinNishinaInteractorPointers kn_params,
             celeritas::SecondaryAllocatorPointers     secondaries,
             celeritas::RngStatePointers               rng_states,
             celeritas::span<celeritas::Real3>         direction,
             celeritas::span<bool>                     alive,
             celeritas::span<celeritas::real_type>     energy_deposition);

//---------------------------------------------------------------------------//
// Sum the total energy deposition
celeritas::real_type
reduce_energy_dep(celeritas::span<celeritas::real_type> edep);

//---------------------------------------------------------------------------//
// Sum the total number of living particles
celeritas::size_type reduce_alive(celeritas::span<bool> alive);

//---------------------------------------------------------------------------//
} // namespace demo_interactor
