//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncCompactedCopy.device.t.hh
//---------------------------------------------------------------------------//
#pragma once

#include "AsyncCompactedCopy.hh"

#include "corecel/Macros.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "DeviceAlgorithms.device.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Select items that evaluate to true
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
    compacted_pending_ = true;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
