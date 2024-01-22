//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/RefImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Store a value/reference and dispatch function name based on MemSpace.

template<class T, MemSpace M>
struct RefGetter
{
    T obj_;

    auto operator()() const -> decltype(auto) { return obj_.host_ref(); }
};

template<class T>
struct RefGetter<T, MemSpace::device>
{
    T obj_;

    auto operator()() const -> decltype(auto) { return obj_.device_ref(); }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
