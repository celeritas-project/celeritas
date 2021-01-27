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
#include "physics/base/ParticleInterface.hh"
#include "physics/base/SecondaryAllocatorInterface.hh"
#include "physics/em/detail/KleinNishina.hh"
#include "physics/grid/XsGridInterface.hh"
#include "random/cuda/RngInterface.hh"
#include "DetectorInterface.hh"

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
    celeritas::ParticleParamsPointers       particle;
    celeritas::XsGridPointers               xs;
    celeritas::detail::KleinNishinaPointers kn_interactor;

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
    celeritas::Span<celeritas::Real3>     position;
    celeritas::Span<celeritas::Real3>     direction;
    celeritas::Span<celeritas::real_type> time;
    celeritas::Span<bool>                 alive;

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
celeritas::size_type reduce_alive(celeritas::Span<bool> alive);

//---------------------------------------------------------------------------//
} // namespace demo_interactor
