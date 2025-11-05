//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreState.cc
//---------------------------------------------------------------------------//
#include "CoreState.hh"

#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/Logger.hh"
#include "corecel/random/params/RngParams.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/random/RngReseed.hh"

#include "CoreParams.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
//! Support polymorphic deletion, anchoring to avoid bugs
CoreStateInterface::~CoreStateInterface() = default;

//---------------------------------------------------------------------------//
//! Default destructor, anchoring to avoid bugs
CoreStateBase::~CoreStateBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct from CoreParams.
 */
template<MemSpace M>
CoreState<M>::CoreState(CoreParams const& params,
                        StreamId stream_id,
                        size_type num_track_slots)
{
    CELER_VALIDATE(stream_id < params.max_streams(),
                   << "stream ID " << stream_id.unchecked_get()
                   << " is out of range: max streams is "
                   << params.max_streams());
    CELER_VALIDATE(num_track_slots > 0, << "number of track slots is not set");

    ScopedProfiling profile_this{"construct-optical-state"};

    states_ = CollectionStateStore<CoreStateData, M>(
        params.host_ref(), stream_id, num_track_slots);

    this->counters().num_vacancies = num_track_slots;

    if constexpr (M == MemSpace::device)
    {
        device_ref_vec_ = DeviceVector<Ref>(1);
        device_ref_vec_.copy_to_device({&this->ref(), 1});
        ptr_ = make_observer(device_ref_vec_);
    }
    else if constexpr (M == MemSpace::host)
    {
        ptr_ = make_observer(&this->ref());
    }

    CELER_LOG(status) << "Initialized Celeritas optical state";
    CELER_ENSURE(states_);
    CELER_ENSURE(ptr_);
}

//---------------------------------------------------------------------------//
// Default destructor
template<MemSpace M>
CoreState<M>::~CoreState() = default;

//---------------------------------------------------------------------------//
/*!
 * Whether the state is being transported with no active particles.
 *
 * The warmup stage is useful for profiling and debugging since the first
 * step iteration can do the following:
 * - Initialize asynchronous memory pools
 * - Interrogate kernel functions for properties to be output later
 * - Allocate "lazy" auxiliary data (e.g. action diagnostics)
 */
template<MemSpace M>
bool CoreState<M>::warming_up() const
{
    CELER_NOT_IMPLEMENTED("warming up");
    return this->counters().num_active == 0;
}

//---------------------------------------------------------------------------//
/*!
 * Inject primaries to be turned into TrackInitializers.
 *
 * These will be converted by the ProcessPrimaries action.
 */
template<MemSpace M>
void CoreState<M>::insert_primaries(Span<TrackInitializer const>)
{
    CELER_NOT_IMPLEMENTED("primary insertion");
}

//---------------------------------------------------------------------------//
/*!
 * Reset the state data.
 *
 * This clears the state counters and initializes the necessary state data so
 * the state can be reused for the next step in the core stepping loop. This
 * should only be necessary if the previous event aborted early.
 */
template<MemSpace M>
void CoreState<M>::reset()
{
    this->counters() = CoreStateCounters{};
    this->counters().num_vacancies = this->size();

    // Reset all the track slots to inactive
    fill(TrackStatus::inactive, &this->ref().sim.status);

    // Mark all the track slots as empty
    fill_sequence(&this->ref().init.vacancies, this->stream_id());
}

//---------------------------------------------------------------------------//
/*!
 * Reseed RNGs at the start of an event for reproducibility.
 *
 * This reinitializes the RNG states using a single seed and unique subsequence
 * for each thread. It ensures that given an event identification, the random
 * number sequence for the event (and thus the event's behavior) can be
 * reproduced.
 */
template<MemSpace M>
void CoreState<M>::reseed(std::shared_ptr<RngParams const> rng,
                          UniqueEventId event_id)
{
    CELER_EXPECT(rng);
    ScopedProfiling profile_this{"reseed"};
    reseed_rng(get_ref<M>(*rng), this->ref().rng, this->stream_id(), event_id);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template class CoreState<MemSpace::host>;
template class CoreState<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
