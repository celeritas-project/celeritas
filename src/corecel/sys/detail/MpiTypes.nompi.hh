//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/MpiTypes.nompi.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct MpiComm
{
    int value_;
};

constexpr inline bool operator==(MpiComm a, MpiComm b)
{
    return a.value_ == b.value_;
}

constexpr inline bool operator!=(MpiComm a, MpiComm b)
{
    return !(a == b);
}

constexpr inline MpiComm MpiCommNull()
{
    return {0};
}
constexpr inline MpiComm MpiCommSelf()
{
    return {-1};
}
constexpr inline MpiComm MpiCommWorld()
{
    return {1};
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
