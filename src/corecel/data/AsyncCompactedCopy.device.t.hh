//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncCompactedCopy.device.t.hh
//---------------------------------------------------------------------------//
#pragma once

#include "AsyncCompactedCopy.hh"

#include <exception>

#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"

#include "Copier.hh"
#include "DeviceAlgorithms.device.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select items that evaluate to true.
 *
 * Unlike \c Identity this returns a \c bool : items may only be explicitly
 * convertible to it, and the device select algorithms (hipCUB and older
 * versions of CUB) assign the result of the predicate to their flags.
 */
struct IsTruthy
{
    template<class T>
    CELER_FUNCTION bool operator()(T const& value) const
    {
        return static_cast<bool>(value);
    }
};
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Enqueue compaction of the valid items on the stream.
 *
 * The device items must remain unchanged until the compaction completes,
 * which is guaranteed if they are only modified by later work on the same
 * stream. The previously compacted items must have been copied.
 */
template<class T>
void AsyncCompactedCopy<T>::compact_async(
    ObserverPtr<T const, MemSpace::device> items, size_type count)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(!this->has_compacted());
    CELER_EXPECT(items || count == 0);
    CELER_EXPECT(count <= this->capacity());

    ScopedProfiling profile_this{"compact-copy"};
    auto& stream = celeritas::device().stream(stream_);

    // Don't overwrite the scratch space while it is being copied
    stream.wait(copied_);

    auto num_selected = make_observer(num_selected_);
    copy_if_async(items.get(),
                  make_observer(scratch_),
                  count,
                  num_selected,
                  detail::IsTruthy{},
                  stream_);
    host_num_selected_.load_async(num_selected, stream_);
    scratch_state_ = ScratchState::compacted;
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

    ScopedProfiling profile_this{"copy-compacted"};

    // The count is loaded after the compaction, so the host-observed load also
    // orders the copy after the compaction
    size_type const count = host_num_selected_.wait();
    scratch_state_ = ScratchState::free;
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
    output_state_ = OutputState::copied;
}

//---------------------------------------------------------------------------//
/*!
 * Wait for any compaction or copy, logging any failure.
 *
 * Waiting on an event that was never recorded returns immediately, so this
 * does not depend on which stages are pending.
 */
template<class T>
void AsyncCompactedCopy<T>::teardown_sync() noexcept
{
    try
    {
        host_num_selected_.sync();
        copied_.sync();
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
}  // namespace celeritas
