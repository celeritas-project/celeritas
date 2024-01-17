//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/KNDemoKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/StackAllocator.hh"
#include "celeritas/em/data/KleinNishinaData.hh"
#include "celeritas/grid/XsGridData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/random/RngData.hh"

#include "DetectorData.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//! Kernel thread dimensions
struct DeviceGridParams
{
    bool sync = false;  //!< Call synchronize after every kernel
};

template<Ownership W, MemSpace M>
struct TableData
{
    template<class T>
    using Items = Collection<T, W, M>;

    Items<real_type> reals;
    XsGridData xs;

    //// MEMBER FUNCTIONS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && xs;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TableData& operator=(TableData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        reals = other.reals;
        xs = other.xs;
        return *this;
    }
};

//! Immutable problem data
template<Ownership W, MemSpace M>
struct ParamsData
{
    ParticleParamsData<W, M> particle;
    RngParamsData<W, M> rng;
    TableData<W, M> tables;
    KleinNishinaData kn_interactor;
    DetectorParamsData detector;

    explicit CELER_FUNCTION operator bool() const
    {
        return particle && tables && kn_interactor && detector;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ParamsData& operator=(ParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        particle = other.particle;
        rng = other.rng;
        tables = other.tables;
        kn_interactor = other.kn_interactor;
        return *this;
    }
};

//! Initial conditions
struct InitialData
{
    ParticleTrackInitializer particle;
};

//! Thread-dependent state data
template<Ownership W, MemSpace M>
struct StateData
{
    using SecondaryAllocatorData = StackAllocatorData<Secondary, W, M>;

    ParticleStateData<W, M> particle;
    RngStateData<W, M> rng;
    Span<Real3> position;
    Span<Real3> direction;
    Span<real_type> time;
    Span<bool> alive;

    SecondaryAllocatorData secondaries;
    DetectorStateData<W, M> detector;

    explicit CELER_FUNCTION operator bool() const
    {
        return particle && rng && secondaries && detector && !position.empty()
               && !direction.empty() && !time.empty() && !alive.empty();
    }

    //! Number of tracks
    CELER_FUNCTION size_type size() const { return particle.size(); }
};

//---------------------------------------------------------------------------//
// Initialize particle states
void initialize(DeviceGridParams const& grid,
                DeviceCRef<ParamsData> const& params,
                DeviceRef<StateData> const& state,
                InitialData const& initial);

//---------------------------------------------------------------------------//
// Run an iteration
void iterate(DeviceGridParams const& grid,
             DeviceCRef<ParamsData> const& params,
             DeviceRef<StateData> const& state);

//---------------------------------------------------------------------------//
// Clean up data
void cleanup(DeviceGridParams const& grid,
             DeviceCRef<ParamsData> const& params,
             DeviceRef<StateData> const& state);

//---------------------------------------------------------------------------//
// Sum the total number of living particles
size_type reduce_alive(DeviceGridParams const& grid, Span<bool const> alive);

//---------------------------------------------------------------------------//
// Finalize, copying tallied data to host
template<MemSpace M>
void finalize(ParamsData<Ownership::const_reference, M> const& params,
              StateData<Ownership::reference, M> const& state,
              Span<real_type> edep)
{
    CELER_EXPECT(edep.size() == params.detector.tally_grid.size);

    copy_to_host(state.detector.tally_deposition, edep);
    real_type const norm = 1 / real_type(state.size());
    for (real_type& v : edep)
    {
        v *= norm;
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
