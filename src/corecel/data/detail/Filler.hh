//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/Filler.hh
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
namespace detail
{
//---------------------------------------------------------------------------//

//! Assign on host or mapped / managed memory
template<class T, MemSpace M>
struct Filler
{
    T const& value;

    void operator()(Span<T> data) const
    {
        std::fill(data.begin(), data.end(), value);
    }
};

//! Assign on device
template<class T>
struct Filler<T, MemSpace::device>
{
  public:
    explicit Filler(T const& value) : value_{value} {};
    Filler(T const& value, StreamId stream)
        : value_{value}, stream_{stream} {};

    void operator()(Span<T>) const;

  private:
    T const& value_;
    StreamId stream_;
};

#if !CELER_USE_DEVICE
template<class T>
void Filler<T, MemSpace::device>::operator()(Span<T>) const
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#else
extern template struct Filler<real_type, MemSpace::device>;
extern template struct Filler<size_type, MemSpace::device>;
extern template struct Filler<int, MemSpace::device>;
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
