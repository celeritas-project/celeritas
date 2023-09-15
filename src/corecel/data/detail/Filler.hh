//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/Filler.hh
//---------------------------------------------------------------------------//
#pragma once

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
template<class T, MemSpace M>
struct Filler;

//! Assign on host
template<class T>
struct Filler<T, MemSpace::host>
{
    T const& value;

    void operator()(Span<T> data) const
    {
        for (T& v : data)
        {
            v = value;
        }
    }
};

//! Assign on device
template<class T>
struct Filler<T, MemSpace::device>
{
    T const& value;

    void operator()(Span<T>) const;
    void operator()(Span<T>, StreamId) const;
};

#if !CELER_USE_DEVICE
template<class T>
void Filler<T, MemSpace::device>::operator()(Span<T>) const
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

template<class T>
void Filler<T, MemSpace::device>::operator()(Span<T>, StreamId) const
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
