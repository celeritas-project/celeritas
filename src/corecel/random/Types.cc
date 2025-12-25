//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a 1D distribution.
 */
char const* to_cstring(OnedDistributionType type)
{
    static EnumStringMapper<OnedDistributionType> const to_cstring_impl{
        "delta",
        "normal",
    };
    return to_cstring_impl(type);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a 3D distribution.
 */
char const* to_cstring(ThreedDistributionType type)
{
    static EnumStringMapper<ThreedDistributionType> const to_cstring_impl{
        "delta",
        "isotropic",
        "uniform box",
    };
    return to_cstring_impl(type);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
