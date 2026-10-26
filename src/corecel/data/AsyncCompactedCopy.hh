//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncCompactedCopy.hh
//---------------------------------------------------------------------------//
#pragma once

#include <exception>
#include <type_traits>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceEvent.hh"
#include "corecel/sys/Stream.hh"

#include "AsyncSnapshot.hh"
#include "Copier.hh"
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
 * waits for each step to complete before starting the next one.
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
 * \c compact_async is defined in \c AsyncCompactedCopy.device.t.hh , which
 * must be included by the device source file that calls it.
 */
template<class T>
class AsyncCompactedCopy
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "Copied items must be trivially copyable");

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

    // Prevent copying (the pinned output is the target of async copies)
    CELER_DELETE_COPY_MOVE(AsyncCompactedCopy);

    //! Maximum number of items
    size_type capacity() const { return scratch_.size(); }

    //! Whether storage is allocated
    explicit operator bool() const { return !scratch_.empty(); }

    //! Whether compacted items are waiting to be copied
    bool has_compacted() const { return compacted_pending_; }

    //! Whether copied items are waiting to be taken
    bool has_copied() const { return copied_pending_; }

    // Enqueue compaction of the valid items on the stream
    void compact_async(ObserverPtr<T const, MemSpace::device> items,
                       size_type count);

    // Enqueue the copy of the compacted items on the memcpy stream
    inline void copy_async();

    // Wait for the copied items and release them to the caller
    inline SpanConstT take_copied();

  private:
    StreamId stream_;
    DeviceVector<T> scratch_;
    DeviceVector<size_type> num_selected_;
    AsyncSnapshot<size_type> host_num_selected_;
    VecT output_;
    DeviceEvent copied_{nullptr};
    bool compacted_pending_{false};
    bool copied_pending_{false};
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
    try
    {
        if (compacted_pending_)
        {
            host_num_selected_.sync();
        }
        if (copied_pending_)
        {
            copied_.sync();
        }
    }
    catch (std::exception const& e)
    {
        CELER_LOG_LOCAL(error) << "Failed to wait for pending compaction or "
                                  "copy while destroying compacted copier: "
                               << e.what();
    }
    catch (...)
    {
        CELER_LOG_LOCAL(error) << "Failed to wait for pending compaction or "
                                  "copy while destroying compacted copier: "
                                  "unknown exception";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue the copy of the compacted items on the memcpy stream.
 *
 * This reads the number of compacted items on the host, waiting for the
 * compaction if it is not yet complete. The previously copied items must have
 * been taken.
 */
template<class T>
void AsyncCompactedCopy<T>::copy_async()
{
    CELER_EXPECT(this->has_compacted());
    CELER_EXPECT(!this->has_copied());

    // The count is loaded after the compaction, so the host-observed load also
    // orders the copy after the compaction
    size_type const count = host_num_selected_.wait();
    compacted_pending_ = false;
    CELER_ASSERT(count <= this->capacity());
    output_.resize(count);

    auto& stream = celeritas::device().memcpy_stream(stream_);
    if (count > 0)
    {
        copy_bytes<MemSpace::host, MemSpace::device>(
            ObserverPtr<T, MemSpace::host>{output_.data()},
            make_observer(scratch_),
            count * sizeof(T),
            stream);
    }
    copied_.record(stream);
    copied_pending_ = true;
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
    copied_.sync();
    copied_pending_ = false;
    return make_span(output_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
