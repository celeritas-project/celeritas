//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/EnumClassUtils.hh
//! \brief Device-friendly utilities for mapping classes to variants
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper struct for mapping enums to classes.
 *
 * This class can be passed as a "tag" to functors that can then retrieve its
 * value or the associated class. It can be implicitly converted into a
 * SurfaceType enum for use in template parameters.
 */
template<class E, E EV, class T>
struct EnumToClass
{
    using enum_type = E;
    using type = T;

    static constexpr enum_type value = EV;

    CELER_CONSTEXPR_FUNCTION operator E() const noexcept { return value; }
    CELER_CONSTEXPR_FUNCTION enum_type operator()() const noexcept
    {
        return value;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
