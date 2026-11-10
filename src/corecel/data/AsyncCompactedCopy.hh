//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncCompactedCopy.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceEvent.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "AsyncSnapshot.hh"
#include "DeviceVector.hh"
#include "ObserverPtr.hh"
#include "PinnedAllocator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Compact valid device items and copy them to the host without waiting.
 *
 * This transfers the items that evaluate to \c true (e.g., detector hits with
 * a valid detector ID) from a per-track-slot device array to pinned host
 * memory in three phases, so that the transfer overlaps with later kernels
 * and the host only reads the number of items once it is available:
 * 1. \b Compacted: \c compact_async enqueues on the given stream the stable
 *    selection of the valid items into a device scratch buffer, followed by
 *    an asynchronous copy of their number to pinned host memory. The
 *    compaction waits on the device for any previous copy out of the
 *    scratch buffer.
 * 2. \b Copied: once the compaction is complete (for example because the host
 *    waited on a later event on the same stream), \c copy_async reads the
 *    number of items and enqueues the copy of exactly that many items to the
 *    pinned host output on the device's \c memcpy_stream .
 * 3. \b Processed: \c take_copied waits for the copy and returns the items,
 *    which remain valid until the next call to \c copy_async .
 *
 * The scratch buffer and host output form a two-stage pipeline: the items of
 * step \em n+1 can be compacted while the items of step \em n are being
 * copied or processed. A single host output therefore suffices if the host
 * waits for each step to complete before starting the next one. Those phases
 * describe one set of items: each buffer has its own state, with the scratch
 * \em free or \em compacted and the output \em empty or \em copied , and any
 * combination of the two can occur.
 *
 * \code
    // At the end of step n
    if (copier.has_compacted())
    {
        process(copier.take_copied());  // step n-2
        copier.copy_async();  // step n-1
    }
    copier.compact_async(device_items.data(), device_items.size());  // step n
   \endcode
 *
 * The device operations are defined in \c AsyncCompactedCopy.device.t.hh : one
 * device source file must include it and explicitly instantiate this class for
 * each item type.
 */
template<class T>
class AsyncCompactedCopy
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "Copied items must be trivially copyable");

    static constexpr bool ndebug = !CELERITAS_DEBUG;

  public:
    //!@{
    //! \name Type aliases
    using SpanConstT = Span<T const>;
    using VecT = std::vector<T, PinnedAllocator<T>>;
    //!@}

  public:
    //! Construct without storage
    AsyncCompactedCopy() = default;

    // Allocate device scratch and pinned host storage for a stream
    inline AsyncCompactedCopy(size_type capacity, StreamId stream);

    // Wait for any pending compaction or copy before releasing buffers
    inline ~AsyncCompactedCopy();

    //!@{
    //! Prevent copying and move construction
    AsyncCompactedCopy(AsyncCompactedCopy const&) = delete;
    AsyncCompactedCopy(AsyncCompactedCopy&&) = delete;
    AsyncCompactedCopy& operator=(AsyncCompactedCopy const&) = delete;
    //!@}

    // Replace the storage, leaving the old storage to be released by the other
    inline AsyncCompactedCopy& operator=(AsyncCompactedCopy&& other) noexcept(
        ndebug);

    //! Maximum number of items
    size_type capacity() const { return scratch_.size(); }

    //! Whether storage is allocated
    explicit operator bool() const { return !scratch_.empty(); }

    //! Whether compacted items are waiting to be copied
    bool has_compacted() const
    {
        return scratch_state_ == ScratchState::compacted;
    }

    //! Whether copied items are waiting to be taken
    bool has_copied() const { return output_state_ == OutputState::copied; }

    // Enqueue compaction of the valid items on the stream
    void compact_async(ObserverPtr<T const, MemSpace::device> items,
                       size_type count);

    // Enqueue the copy of the compacted items on the memcpy stream
    void copy_async();

    // Wait for the copied items and release them to the caller
    inline SpanConstT take_copied();

  private:
    //! State of the device scratch space (see the class documentation)
    enum class ScratchState
    {
        free,  //!< Available for the next compaction
        compacted,  //!< Holds compacted items awaiting a copy
    };

    //! State of the pinned host output (see the class documentation)
    enum class OutputState
    {
        empty,  //!< Available for the next copy
        copied,  //!< Holds copied items awaiting processing
    };

    StreamId stream_;
    DeviceVector<T> scratch_;
    DeviceVector<size_type> num_selected_;
    AsyncSnapshot<size_type> host_num_selected_;
    VecT output_;
    DeviceEvent copied_{nullptr};
    ScratchState scratch_state_{ScratchState::free};
    OutputState output_state_{OutputState::empty};

    // Wait for any compaction or copy, logging any failure
    void teardown_sync() noexcept;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Allocate device scratch and pinned host storage for a stream.
 */
template<class T>
AsyncCompactedCopy<T>::AsyncCompactedCopy(size_type capacity, StreamId stream)
    : stream_{stream}
    , scratch_(capacity, stream)
    , num_selected_(1, stream)
    , copied_{celeritas::device()}
{
    CELER_EXPECT(capacity > 0);
    CELER_EXPECT(stream);
    output_.reserve(capacity);
}

//---------------------------------------------------------------------------//
/*!
 * Replace the storage, leaving the old storage to be released by the other.
 *
 * All items must have been taken, since any others would never reach the
 * caller. If they haven't, in a release build the items are dropped, but the
 * old storage is still safe from pending work: it is swapped into \c other ,
 * whose destructor waits for any compaction or copy before releasing it. Any
 * span previously returned by \c take_copied is invalidated.
 */
template<class T>
auto AsyncCompactedCopy<T>::operator=(AsyncCompactedCopy&& other) noexcept(
    ndebug) -> AsyncCompactedCopy&
{
    CELER_EXPECT(!this->has_compacted() && !this->has_copied());

    using std::swap;
    swap(stream_, other.stream_);
    swap(scratch_, other.scratch_);
    swap(num_selected_, other.num_selected_);
    swap(host_num_selected_, other.host_num_selected_);
    swap(output_, other.output_);
    swap(copied_, other.copied_);
    swap(scratch_state_, other.scratch_state_);
    swap(output_state_, other.output_state_);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Wait for any pending compaction or copy before releasing buffers.
 *
 * The number of compacted items and the copied items are written
 * asynchronously into pinned memory owned by this class, so pending copies
 * must complete before that memory is freed (e.g., when an exception unwinds
 * the transport loop mid-step).
 */
template<class T>
AsyncCompactedCopy<T>::~AsyncCompactedCopy()
{
    if (*this)
    {
        this->teardown_sync();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Wait for the copied items and release them to the caller.
 *
 * The returned items are valid until the next call to \c copy_async . If no
 * items are being copied, an empty span is returned.
 */
template<class T>
auto AsyncCompactedCopy<T>::take_copied() -> SpanConstT
{
    if (!this->has_copied())
    {
        return {};
    }
    ScopedProfiling profile_this{"take-copied"};
    copied_.sync();
    output_state_ = OutputState::empty;
    // make_span is callable from device code, but the vector's accessors aren't
    return {output_.data(), output_.size()};
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<class T>
void AsyncCompactedCopy<T>::copy_async()
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

template<class T>
void AsyncCompactedCopy<T>::teardown_sync() noexcept
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
