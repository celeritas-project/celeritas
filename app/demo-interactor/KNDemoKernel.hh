//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Pie.hh"
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
using celeritas::MemSpace;
using celeritas::Ownership;

//---------------------------------------------------------------------------//
//! Kernel thread dimensions
struct CudaGridParams
{
    unsigned int block_size = 256; //!< Threads per block
    unsigned int grid_size  = 32;  //!< Blocks per grid
    bool         sync       = false; //!< Call synchronize after every kernel
};

template<Ownership W, MemSpace M>
struct TableData
{
    template<class T>
    using Data = celeritas::Pie<T, W, M>;

    Data<celeritas::real_type> reals;
    celeritas::XsGridData      xs;

    //// MEMBER FUNCTIONS ////

    //! Whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && xs;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TableData& operator=(const TableData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        reals = other.reals;
        xs    = other.xs;
        return *this;
    }
};

//! Pointers to immutable problem data
template<Ownership W, MemSpace M>
struct ParamsData
{
    celeritas::ParticleParamsData<W, M>     particle;
    TableData<W, M>                         tables;
    celeritas::detail::KleinNishinaPointers kn_interactor;

    explicit CELER_FUNCTION operator bool() const
    {
        return particle && tables && kn_interactor;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ParamsData& operator=(const ParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        particle      = other.particle;
        tables        = other.tables;
        kn_interactor = other.kn_interactor;
        return *this;
    }
};

using ParamsHostRef = ParamsData<Ownership::const_reference, MemSpace::host>;
using ParamsDeviceRef
    = ParamsData<Ownership::const_reference, MemSpace::device>;

//! Pointers to initial conditions
struct InitialPointers
{
    celeritas::ParticleTrackState particle;
};

//! Pointers to thread-dependent state data
template<Ownership W, MemSpace M>
struct StateData
{
    celeritas::ParticleStateData<W, M>    particle;
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

using StateHostRef   = StateData<Ownership::reference, MemSpace::host>;
using StateDeviceRef = StateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
// Initialize particle states
void initialize(const CudaGridParams&  grid,
                const ParamsDeviceRef& params,
                const StateDeviceRef&  state,
                const InitialPointers& initial);

//---------------------------------------------------------------------------//
// Run an iteration
void iterate(const CudaGridParams&                        grid,
             const ParamsDeviceRef&                       params,
             const StateDeviceRef&                        state,
             const celeritas::SecondaryAllocatorPointers& secondaries,
             const celeritas::DetectorPointers&           detector);

//---------------------------------------------------------------------------//
// Sum the total number of living particles
celeritas::size_type
reduce_alive(celeritas::Span<bool> alive, const CudaGridParams& grid);

//---------------------------------------------------------------------------//
} // namespace demo_interactor
