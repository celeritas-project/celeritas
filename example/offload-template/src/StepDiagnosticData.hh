//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/StepDiagnosticData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
//! Step statistics gathered inside a kernel
struct NativeStepStatistics
{
    using real_type = double;

    real_type step_length{};
    real_type energy_deposited{};
};

//---------------------------------------------------------------------------//
//! Step statistics gathered in host memory
struct HostStepStatistics
{
    //! Accumulated number of steps
    size_type steps{};

    //! Accumulated number of new tracks
    size_type generated{};

    //! Accumulated number of secondaries
    size_type secondaries{};
};

//---------------------------------------------------------------------------//
/*!
 * Manage data ownership and reference and transfer between host/device.
 *
 * Only some of the data (\c NativeStepStatistics) is updated on device. The
 * number of steps must be changed outside the kernel.
 */
template<Ownership W, MemSpace M>
struct StepStateData
{
    //// TYPES ////

    using size_type = unsigned long long;
    template<class T>
    using Items = celeritas::Collection<T, W, M>;

    //// DATA ////

    //! Accumulated data (one per simultaneous event, currently fixed at 1)
    Items<NativeStepStatistics> data;

    //! Accumulated data on host
    HostStepStatistics host_data;

    //// METHODS ////

    //! True if constructed and correctly sized
    explicit CELER_FUNCTION operator bool() const { return data.size() > 0; }

    //! State size (number of events)
    CELER_FUNCTION TrackSlotId::size_type size() const { return data.size(); }

    //! Assign (including H<->D transfer) from another set of states
    template<Ownership W2, MemSpace M2>
    StepStateData& operator=(StepStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        data = other.data;
        host_data = other.host_data;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Allocate step diagnostic data.
 *
 * Since we only have one event in flight for Geant4 integration, the size will
 * be one. Altering this will require additional extension (device "params"
 * that store the number of events). The stream ID (second argument,
 * corresponding to worker thread index) and size (the number of track slots)
 * are not needed for this constructor.
 */
template<MemSpace M>
inline void
resize(StepStateData<Ownership::value, M>* state, StreamId, size_type)
{
    CELER_EXPECT(state);
    resize(&state->data, 1);
}

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
