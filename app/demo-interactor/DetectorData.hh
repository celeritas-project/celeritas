//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/DetectorData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/StackAllocatorData.hh"
#include "corecel/grid/UniformGridData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/data/CollectionBuilder.hh"
#endif

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Energy deposition event in the detector.
 *
 * Note that most of the data is discarded at integration time.
 */
struct Hit
{
    Real3 pos;
    Real3 dir;
    TrackSlotId track_slot;
    real_type time;
    units::MevEnergy energy_deposited;
};

//---------------------------------------------------------------------------//
/*!
 * Interface to detector hit buffer.
 */
struct DetectorParamsData
{
    UniformGridData tally_grid;

    //! Whether the data is initialized
    explicit CELER_FUNCTION operator bool() const { return bool(tally_grid); }
};

//---------------------------------------------------------------------------//
/*!
 * Interface to detector hit buffer.
 */
template<Ownership W, MemSpace M>
struct DetectorStateData
{
    StackAllocatorData<Hit, W, M> hit_buffer;
    Collection<real_type, W, M> tally_deposition;

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return bool(hit_buffer) && !tally_deposition.empty();
    }

    //! Total capacity of hit buffer
    CELER_FUNCTION size_type capacity() const { return hit_buffer.capacity(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DetectorStateData& operator=(DetectorStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        hit_buffer = other.hit_buffer;
        tally_deposition = other.tally_deposition;
        return *this;
    }
};

#if !CELER_DEVICE_COMPILE
//---------------------------------------------------------------------------//
/*!
 * Allocate components and capacity for the detector.
 */
template<MemSpace M>
inline void resize(DetectorStateData<Ownership::value, M>* data,
                   DetectorParamsData const& params,
                   size_type size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    resize(&data->hit_buffer, size);
    resize(&data->tally_deposition, params.tally_grid.size);
}
#endif

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
