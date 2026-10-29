//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorCounterState.cc
//---------------------------------------------------------------------------//
#include "GeneratorCounterState.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionBuilder.hh"

#include "detail/GeneratorCounterAlgorithms.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the stream used to update the counters.
 *
 * Like the snapshot of the counters, this must be constructed after the
 * device is activated.
 */
template<MemSpace M>
GeneratorCounterState<M>::GeneratorCounterState(StreamId stream)
    : stream_{stream}
{
    resize(&counters_, 1);
    detail::assign_counters(this->device_counters(), CountersT{}, stream_);
    detail::clear_accum(this->device_counters(), stream_);
}

//---------------------------------------------------------------------------//
/*!
 * Set the current counters.
 *
 * This updates the host copy immediately and enqueues the update of the
 * state's copy, followed by a copy of the updated counters so that the bound
 * is only tightened from the assigned value. The buffer bound is set to the
 * new buffer size.
 */
template<MemSpace M>
void GeneratorCounterState<M>::assign(CountersT const& value)
{
    detail::assign_counters(this->device_counters(), value, stream_);
    snapshot_.load_async(this->device_counters(), stream_);
    this->counters = value;
    bound_ = value.buffer_size;
    growth_ = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Get a conservative upper bound on the current buffer size.
 *
 * If the most recent copy of the counters has completed, the bound is
 * tightened using the copied buffer size and the growth since the copy was
 * enqueued.
 */
template<MemSpace M>
size_type GeneratorCounterState<M>::buffer_bound()
{
    if (snapshot_.ready())
    {
        bound_ = snapshot_.value().current.buffer_size + growth_;
    }
    return bound_;
}

//---------------------------------------------------------------------------//
/*!
 * Increase the buffer bound by the maximum number of items added.
 */
template<MemSpace M>
void GeneratorCounterState<M>::grow_bound(size_type count)
{
    bound_ += count;
    growth_ += count;
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue a copy of the counters used to tighten the buffer bound.
 *
 * This should be called after all the kernels that modify the buffer size in
 * a step have been enqueued. A new copy is not enqueued while a previous one
 * is still pending, since the host could not use it any earlier.
 */
template<MemSpace M>
void GeneratorCounterState<M>::copy_async()
{
    if (!snapshot_.ready())
    {
        return;
    }
    snapshot_.load_async(this->device_counters(), stream_);
    growth_ = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Synchronize and update the host counters and statistics.
 *
 * The increments accumulated in the state's memory space are added to the
 * accumulated statistics and cleared.
 */
template<MemSpace M>
void GeneratorCounterState<M>::sync_counters()
{
    snapshot_.load_async(this->device_counters(), stream_);
    auto const& value = snapshot_.wait();
    this->counters = value.current;
    this->accum.buffer_size += value.accum.buffer_size;
    this->accum.num_pending += value.accum.num_pending;
    this->accum.num_generated += value.accum.num_generated;
    detail::clear_accum(this->device_counters(), stream_);

    bound_ = value.current.buffer_size;
    growth_ = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Clear the generated count at the start of a transport loop.
 */
template<MemSpace M>
void GeneratorCounterState<M>::clear_generated()
{
    detail::clear_generated(this->device_counters(), stream_);
    this->counters.num_generated = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Discard the current counters after the loop aborted early.
 *
 * The accumulated increments are kept.
 */
template<MemSpace M>
void GeneratorCounterState<M>::reset_counters()
{
    this->assign(CountersT{});
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class GeneratorCounterState<MemSpace::host>;
template class GeneratorCounterState<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
