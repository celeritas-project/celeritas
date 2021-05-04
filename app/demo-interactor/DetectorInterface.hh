//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DetectorInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/OpaqueId.hh"
#include "base/StackAllocatorInterface.hh"
#include "physics/grid/UniformGridInterface.hh"
#include "physics/base/Units.hh"

#ifndef __CUDA_ARCH__
#    include "base/CollectionBuilder.hh"
#endif

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Energy deposition event in the detector.
 *
 * Note that most of the data is discarded at integration time.
 */
struct Hit
{
    celeritas::Real3            pos;
    celeritas::Real3            dir;
    celeritas::ThreadId         thread;
    celeritas::real_type        time;
    celeritas::units::MevEnergy energy_deposited;
};

//---------------------------------------------------------------------------//
/*!
 * Interface to detector hit buffer.
 */
struct DetectorParamsData
{
    celeritas::UniformGridData tally_grid;

    //! Whether the data is initialized
    explicit CELER_FUNCTION operator bool() const { return bool(tally_grid); }
};

//---------------------------------------------------------------------------//
/*!
 * Interface to detector hit buffer.
 */
template<celeritas::Ownership W, celeritas::MemSpace M>
struct DetectorStateData
{
    using real_type = celeritas::real_type;
    using size_type = celeritas::size_type;

    celeritas::StackAllocatorData<Hit, W, M> hit_buffer;
    celeritas::Collection<real_type, W, M>   tally_deposition;

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return bool(hit_buffer) && !tally_deposition.empty();
    }

    //! Total capacity of hit buffer
    CELER_FUNCTION size_type capacity() const { return hit_buffer.capacity(); }

    //! Assign from another set of data
    template<celeritas::Ownership W2, celeritas::MemSpace M2>
    DetectorStateData& operator=(DetectorStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        hit_buffer       = other.hit_buffer;
        tally_deposition = other.tally_deposition;
        return *this;
    }
};

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
/*!
 * Allocate components and capacity for the detector.
 */
template<celeritas::MemSpace M>
inline void resize(DetectorStateData<celeritas::Ownership::value, M>* data,
                   const DetectorParamsData&                          params,
                   celeritas::size_type                               size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    resize(&data->hit_buffer, size);
    make_builder(&data->tally_deposition).resize(params.tally_grid.size);
}
#endif

//---------------------------------------------------------------------------//
} // namespace demo_interactor
