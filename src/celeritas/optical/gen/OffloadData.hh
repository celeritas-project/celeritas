//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/OffloadData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/GeneratorCounters.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Cumulative statistics of optical tracking.
 */
struct OpticalAccumStats
{
    using size_type = std::size_t;
    using OpticalBufferSize = GeneratorCounters<size_type>;

    std::vector<OpticalBufferSize> generators;

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
