//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Filler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fill contiguous data with copies of a scalar value.
 */
template<class T, MemSpace M>
class Filler
{
  public:
    //! Construct with target value and default stream
    explicit Filler(T value) : value_{value} {}

    //! Construct with target value and specific stream
    Filler(T value, StreamId stream) : value_{value}, stream_{stream} {}

    // Fill the span with the stored value
    inline void operator()(Span<T> data) const;

  private:
    T value_;
    StreamId stream_;

    void fill_device_impl(Span<T> data) const;
};

//---------------------------------------------------------------------------//
/*!
 * Fill the span with the stored value.
 */
template<class T, MemSpace M>
void Filler<T, M>::operator()(Span<T> data) const
{
    if constexpr (M == MemSpace::device)
    {
        this->fill_device_impl(data);
    }
    else
    {
        std::fill(data.begin(), data.end(), value_);
    }
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<class T, MemSpace M>
CELER_FORCEINLINE void Filler<T, M>::fill_device_impl(Span<T>) const
{
    CELER_DISCARD(stream_);
    CELER_ASSERT_UNREACHABLE();
}
#else
extern template class Filler<real_type, MemSpace::device>;
extern template class Filler<size_type, MemSpace::device>;
extern template class Filler<int, MemSpace::device>;
extern template class Filler<TrackSlotId, MemSpace::device>;
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
