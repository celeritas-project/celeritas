//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/UnitUtils.hh
//! \brief Helpers for unit trait classes
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Value is 1 / C1::value()
template<class C1>
struct UnitInverse
{
    //! Get the conversion factor of the resulting unit
    static CELER_CONSTEXPR_FUNCTION auto value() noexcept -> decltype(auto)
    {
        return 1 / C1::value();
    }
};

//---------------------------------------------------------------------------//
//! Value is C1::value() / C2::value()
template<class C1, class C2>
struct UnitDivide
{
    //! Get the conversion factor of the resulting unit
    static CELER_CONSTEXPR_FUNCTION auto value() noexcept -> decltype(auto)
    {
        return C1::value() / C2::value();
    }
};

//! Value is C1::value() * C2::value()
template<class C1, class C2>
struct UnitProduct
{
    //! Get the conversion factor of the resulting unit
    static CELER_CONSTEXPR_FUNCTION auto value() noexcept -> decltype(auto)
    {
        return C1::value() * C2::value();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
