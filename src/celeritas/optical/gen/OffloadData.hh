//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/OffloadData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Current sizes of the buffers of distribution data.
 *
 * These sizes are updated by value on the host at each core step. To allow
 * accumulation over many steps which each may have many photons, the type is
 * templated.
 */
template<class T = ::celeritas::size_type>
struct OpticalOffloadCounters
{
    using size_type = T;

    //! Number of generators
    size_type distributions{0};
    //! Number of generated tracks
    size_type photons{0};

    //! True if any queued generators/tracks exist
    CELER_FUNCTION bool empty() const
    {
        return distributions == 0 && photons == 0;
    }
};

//! Cumulative statistics of optical tracking
struct OpticalAccumStats
{
    using size_type = std::size_t;
    using OpticalBufferSize = OpticalOffloadCounters<size_type>;

    OpticalBufferSize cherenkov;
    OpticalBufferSize scintillation;

    size_type steps{0};
    size_type step_iters{0};
    size_type flushes{0};
};

//---------------------------------------------------------------------------//
/*!
 * Pre-step data needed to generate optical photon distributions.
 *
 * If the optical material is not set, the other properties are invalid.
 */
struct OffloadPreStepData
{
    units::LightSpeed speed;
    Real3 pos{};
    real_type time{};
    OptMatId material;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return material && speed > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Pre-step data that is cached and used to generate optical distributions.
 */
template<Ownership W, MemSpace M>
struct OffloadStepStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    // Pre-step data for generating optical photon distributions
    StateItems<OffloadPreStepData> step;

    //// METHODS ////

    //! Number of states
    CELER_FUNCTION size_type size() const { return step.size(); }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const { return !step.empty(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OffloadStepStateData& operator=(OffloadStepStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        step = other.step;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize optical step states.
 */
template<MemSpace M>
void resize(OffloadStepStateData<Ownership::value, M>* state,
            StreamId,
            size_type size)
{
    CELER_EXPECT(size > 0);
    resize(&state->step, size);
    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
