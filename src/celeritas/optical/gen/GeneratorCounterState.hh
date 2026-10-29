//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorCounterState.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/AsyncSnapshot.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/ObserverPtr.hh"
#include "celeritas/phys/GeneratorCounters.hh"
#include "celeritas/phys/GeneratorInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Generator counters that are updated in the state's memory space.
 *
 * The counters are modified by kernels during stepping so that the host never
 * has to wait on the device to update them. The host copy in
 * \c GeneratorStateBase::counters is only refreshed by \c sync_counters ,
 * which should be called once stepping has completed.
 *
 * For buffers that are filled during stepping, this class also tracks a
 * conservative upper bound on the buffer size so that capacity can be
 * validated and fixed-size device algorithms can be launched without
 * synchronizing. The bound is grown by the maximum number of items that may
 * be added in each step, and it is tightened whenever an asynchronous copy of
 * the counters, enqueued with \c copy_async, has completed.
 *
 * Distributions and photons are counted in the accumulated statistics when
 * they are queued, and photons again when they are generated. Since these
 * counts are also updated by kernels, the increments are kept in the state's
 * memory and added to \c GeneratorStateBase::accum by \c sync_counters .
 */
template<MemSpace M>
class GeneratorCounterState : public GeneratorStateBase
{
  public:
    //!@{
    //! \name Type aliases
    using CountersT = GeneratorCounters<size_type>;
    using DevicePtr = ObserverPtr<GeneratorStateCounters, M>;
    //!@}

  public:
    // Construct with the stream used to update the counters
    explicit GeneratorCounterState(StreamId stream);

    //! Stream used to update the counters
    StreamId stream_id() const { return stream_; }

    //! Counters in the state's memory space
    DevicePtr device_counters() { return counters_.data(); }

    // Set the current counters (host and state copies)
    void assign(CountersT const& value);

    // Get a conservative upper bound on the current buffer size
    size_type buffer_bound();

    // Increase the buffer bound by the maximum number of items added
    void grow_bound(size_type count);

    // Enqueue a copy of the counters used to tighten the buffer bound
    void copy_async();

    // Synchronize and update the host counters and statistics
    void sync_counters() override;

    // Clear the generated count at the start of a transport loop
    void clear_generated() override;

    // Discard the current counters after the loop aborted early
    void reset_counters() override;

  private:
    StreamId stream_;
    Collection<GeneratorStateCounters, Ownership::value, M> counters_;
    AsyncSnapshot<GeneratorStateCounters> snapshot_;

    // Upper bound on the buffer size, and growth since the most recent copy
    size_type bound_{0};
    size_type growth_{0};
};

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class GeneratorCounterState<MemSpace::host>;
extern template class GeneratorCounterState<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
