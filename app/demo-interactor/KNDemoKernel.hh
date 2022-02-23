//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/CollectionAlgorithms.hh"
#include "base/Span.hh"
#include "base/StackAllocator.hh"
#include "base/Types.hh"
#include "physics/base/ParticleData.hh"
#include "physics/base/Secondary.hh"
#include "physics/em/detail/KleinNishinaData.hh"
#include "physics/grid/XsGridData.hh"
#include "random/RngData.hh"

#include "DetectorData.hh"

namespace demo_interactor
{
//---------------------------------------------------------------------------//
using celeritas::MemSpace;
using celeritas::Ownership;

//---------------------------------------------------------------------------//
//! Kernel thread dimensions
struct DeviceGridParams
{
    unsigned int block_size = 256;   //!< Threads per block
    bool         sync       = false; //!< Call synchronize after every kernel
};

template<Ownership W, MemSpace M>
struct TableData
{
    template<class T>
    using Items = celeritas::Collection<T, W, M>;

    Items<celeritas::real_type> reals;
    celeritas::XsGridData       xs;

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

//! Immutable problem data
template<Ownership W, MemSpace M>
struct ParamsData
{
    celeritas::ParticleParamsData<W, M> particle;
    TableData<W, M>                     tables;
    celeritas::detail::KleinNishinaData kn_interactor;
    DetectorParamsData                  detector;

    explicit CELER_FUNCTION operator bool() const
    {
        return particle && tables && kn_interactor && detector;
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

//! Initial conditions
struct InitialData
{
    celeritas::ParticleTrackState particle;
};

//! Thread-dependent state data
template<Ownership W, MemSpace M>
struct StateData
{
    using SecondaryAllocatorData
        = celeritas::StackAllocatorData<celeritas::Secondary, W, M>;

    celeritas::ParticleStateData<W, M>                              particle;
    celeritas::RngStateData<Ownership::reference, MemSpace::device> rng;
    celeritas::Span<celeritas::Real3>                               position;
    celeritas::Span<celeritas::Real3>                               direction;
    celeritas::Span<celeritas::real_type>                           time;
    celeritas::Span<bool>                                           alive;

    SecondaryAllocatorData  secondaries;
    DetectorStateData<W, M> detector;

    explicit CELER_FUNCTION operator bool() const
    {
        return particle && rng && secondaries && detector && !position.empty()
               && !direction.empty() && !time.empty() && !alive.empty();
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
void initialize(const DeviceGridParams& grid,
                const ParamsDeviceRef&  params,
                const StateDeviceRef&   state,
                const InitialData&      initial);

//---------------------------------------------------------------------------//
// Run an iteration
void iterate(const DeviceGridParams& grid,
             const ParamsDeviceRef&  params,
             const StateDeviceRef&   state);

//---------------------------------------------------------------------------//
// Clean up data
void cleanup(const DeviceGridParams& grid,
             const ParamsDeviceRef&  params,
             const StateDeviceRef&   state);

//---------------------------------------------------------------------------//
// Sum the total number of living particles
celeritas::size_type
reduce_alive(const DeviceGridParams& grid, celeritas::Span<const bool> alive);

//---------------------------------------------------------------------------//
// Finalize, copying tallied data to host
template<celeritas::MemSpace M>
void finalize(const ParamsData<Ownership::const_reference, M>& params,
              const StateData<Ownership::reference, M>&        state,
              celeritas::Span<celeritas::real_type>            edep)
{
    CELER_EXPECT(edep.size() == params.detector.tally_grid.size);
    using celeritas::real_type;

    celeritas::copy_to_host(state.detector.tally_deposition, edep);
    const real_type norm = 1 / real_type(state.size());
    for (real_type& v : edep)
    {
        v *= norm;
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
