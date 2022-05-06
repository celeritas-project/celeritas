//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/Filler.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/Types.hh"

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
    const T& value;

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
    const T& value;

    void operator()(Span<T>) const;
};

#if !CELER_USE_DEVICE
template<class T>
void Filler<T, MemSpace::device>::operator()(Span<T>) const
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
