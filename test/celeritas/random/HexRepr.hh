//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/HexRepr.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

template<class T>
struct HexRepr
{
    T value;
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, HexRepr<T> const& h)
{
    ScopedStreamFormat save_fmt(&os);

    os << std::hexfloat << h.value;
    return os;
}

template<class T>
inline HexRepr<T> hex_repr(T value)
{
    return HexRepr<T>{value};
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
