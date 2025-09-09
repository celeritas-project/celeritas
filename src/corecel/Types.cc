//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "io/EnumStringMapper.hh"
#include "io/StringEnumMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a memory space.
 */
char const* to_cstring(MemSpace value)
{
    static EnumStringMapper<MemSpace> const to_cstring_impl{
        "host", "device", "mapped"};
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a unit system.
 */
char const* to_cstring(UnitSystem value)
{
    static_assert(static_cast<int>(UnitSystem::cgs) == CELERITAS_UNITS_CGS);
    static_assert(static_cast<int>(UnitSystem::si) == CELERITAS_UNITS_SI);
    static_assert(static_cast<int>(UnitSystem::clhep) == CELERITAS_UNITS_CLHEP);
    static_assert(static_cast<int>(UnitSystem::native) == CELERITAS_UNITS);

    static EnumStringMapper<UnitSystem> const to_cstring_impl{
        "none", "cgs", "si", "clhep"};
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a unit system corresponding to a string value.
 */
UnitSystem to_unit_system(std::string const& s)
{
    static auto const from_string
        = StringEnumMapper<UnitSystem>::from_cstring_func(to_cstring,
                                                          "unit system");
    return from_string(s);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
