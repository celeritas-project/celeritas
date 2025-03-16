//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/StepDiagnosticData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <celeritas/Quantities.hh>
#include <celeritas/Types.hh>
#include <celeritas/Units.hh>
#include <corecel/Macros.hh>
#include <corecel/data/Collection.hh>
#include <corecel/data/CollectionAlgorithms.hh>
#include <corecel/data/CollectionBuilder.hh>
#include <corecel/data/Filler.hh>

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
    real_type energy_deposition{};  // MeV
};
}  // namespace example

#if CELER_USE_DEVICE
// The Filler is instantiated in StepDiagnosticData.cu
extern template class Filler<example::NativeStepStatistics, MemSpace::device>;
#endif

namespace example
{
//---------------------------------------------------------------------------//
//! Step statistics gathered in host memory
struct HostStepStatistics
{
    //! Accumulated number of steps (use an extra-long int)
    unsigned long long steps{};

    //! Accumulated number of new tracks
    size_type generated{};

    //! Accumulated number of secondaries
    size_type secondaries{};
};

//---------------------------------------------------------------------------//
/*!
 * Shared setup data.
 *
 * This class is where problem setup data, including variable-length data that
 * gets copied to device, is defined.
 */
template<Ownership W, MemSpace M>
struct StepParamsData
{
    static constexpr EventId::size_type num_events{1};

    //! Whether the class is non-empty
    explicit CELER_FUNCTION operator bool() const { return true; }

    //! Assign from another set of data (null-op since no member data)
    template<Ownership W2, MemSpace M2>
    StepParamsData& operator=(StepParamsData<W2, M2> const&)
    {
        return *this;
    }
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

    template<class T>
    using EventItems = celeritas::Collection<T, W, M, EventId>;

    //// DATA ////

    //! Accumulated data (one per simultaneous event, currently fixed at 1)
    EventItems<NativeStepStatistics> data;

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
 * Clear step diagnostic data.
 */
template<Ownership W, MemSpace M>
inline void reset(StepStateData<W, M>* step_state, StreamId sid)
{
    celeritas::fill(NativeStepStatistics{0, 0}, &step_state->data);
    step_state->host_data = {};
}

//---------------------------------------------------------------------------//
/*!
 * Allocate step diagnostic data.
 *
 * Since we only have one event in flight for Geant4 integration, the size will
 * be one. Altering this will require additional extension (device "params"
 * that store the number of events). The stream ID (second argument,
 * corresponding to worker thread index) and size (the number of track slots)
 * are not needed for this constructor.
 *
 * This class is called under the hood by \c celeritas::make_aux_state .
 */
template<MemSpace M>
inline void resize(StepStateData<Ownership::value, M>* state,
                   HostCRef<StepParamsData> const& params,
                   StreamId sid,
                   size_type)
{
    CELER_EXPECT(state);
    resize(&state->data, params.num_events);
    reset(state, sid);
}

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
