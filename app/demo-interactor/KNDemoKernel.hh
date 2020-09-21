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
#include "random/cuda/RngStatePointers.hh"
#include "PhysicsArrayPointers.hh"
#include "DetectorPointers.hh"

namespace demo_interactor
{
//---------------------------------------------------------------------------//
//! Kernel thread dimensions
struct CudaGridParams
{
    unsigned int block_size = 256; //!< Threads per block
    unsigned int grid_size  = 32;  //!< Blocks per grid
};

//! Pointers to immutable problem data
struct ParamPointers
{
    celeritas::ParticleParamsPointers         particle;
    celeritas::PhysicsArrayPointers           xs;
    celeritas::KleinNishinaInteractorPointers kn_interactor;

    explicit CELER_FUNCTION operator bool() const
    {
        return particle && xs && kn_interactor;
    }
};

//! Pointers to initial conditoins
struct InitialPointers
{
    celeritas::ParticleTrackState particle;
};

//! Pointers to thread-dependent state data
struct StatePointers
{
    celeritas::ParticleStatePointers      particle;
    celeritas::RngStatePointers           rng;
    celeritas::span<celeritas::Real3>     position;
    celeritas::span<celeritas::Real3>     direction;
    celeritas::span<celeritas::real_type> time;
    celeritas::span<bool>                 alive;

    explicit CELER_FUNCTION operator bool() const
    {
        return particle && rng && !position.empty() && !direction.empty()
               && !time.empty() && !alive.empty();
    }

    //! Number of tracks
    CELER_FUNCTION celeritas::size_type size() const
    {
        return particle.size();
    }
};

//---------------------------------------------------------------------------//
// Initialize particle states
void initialize(const CudaGridParams&  grid,
                const ParamPointers&   params,
                const StatePointers&   state,
                const InitialPointers& initial);

//---------------------------------------------------------------------------//
// Run an iteration
void iterate(const CudaGridParams&                        grid,
             const ParamPointers&                         params,
             const StatePointers&                         state,
             const celeritas::SecondaryAllocatorPointers& secondaries,
             const celeritas::DetectorPointers&           detector);

//---------------------------------------------------------------------------//
// Sum the total number of living particles
celeritas::size_type reduce_alive(celeritas::span<bool> alive);

//---------------------------------------------------------------------------//
} // namespace demo_interactor
