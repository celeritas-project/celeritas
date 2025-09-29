//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportUnits.cc
//---------------------------------------------------------------------------//
#include "ImportUnits.hh"

#include <type_traits>

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/UnitTypes.hh"

using celeritas::units::visit_unit_system;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Get the length scale of a unit system.
 */
double length_scale(UnitSystem sys)
{
    return visit_unit_system(
        [](auto traits) {
            using Unit = typename decltype(traits)::Length;
            return native_value_from(Quantity<Unit, double>{1});
        },
        sys);
}

//---------------------------------------------------------------------------//
/*!
 * Get the time scale of a unit system.
 */
double time_scale(UnitSystem sys)
{
    return visit_unit_system(
        [](auto traits) {
            using Unit = typename decltype(traits)::Time;
            return native_value_from(Quantity<Unit, double>{1});
        },
        sys);
}

//---------------------------------------------------------------------------//
/*!
 * Get the mass scale of a unit system.
 */
double mass_scale(UnitSystem sys)
{
    return visit_unit_system(
        [](auto traits) {
            using Unit = typename decltype(traits)::Mass;
            return native_value_from(Quantity<Unit, double>{1});
        },
        sys);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Get a printable label for units.
 */
char const* to_cstring(ImportUnits value)
{
    static EnumStringMapper<ImportUnits> const to_cstring_impl{
        "unitless",
        "MeV",
        "MeV/len",
        "len",
        "1/len",
        "1/len-MeV",
        "MeV^2/len",
        "len^2",
        "MeV-len^2",
        "time",
        "1/len^3",
        "len-time^2/mass",
        "1/MeV",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get the native value from a quantity of this type.
 */
double native_value_from(UnitSystem sys, ImportUnits q)
{
    constexpr double mev = 1;
    double const len = length_scale(sys);
    double const time = time_scale(sys);

    switch (q)
    {
        case ImportUnits::unitless:
            return 1;
        case ImportUnits::mev:
            return mev;
        case ImportUnits::inv_mev:
            return 1 / mev;
        case ImportUnits::mev_per_len:
            return mev / len;
        case ImportUnits::len:
            return len;
        case ImportUnits::len_inv:
            return 1 / len;
        case ImportUnits::len_mev_inv:
            return 1 / (len * mev);
        case ImportUnits::mev_sq_per_len:
            return ipow<2>(mev) / len;
        case ImportUnits::len_sq:
            return ipow<2>(len);
        case ImportUnits::mev_len_sq:
            return mev * ipow<2>(len);
        case ImportUnits::time:
            return time;
        case ImportUnits::inv_len_cb:
            return 1 / ipow<3>(len);
        case ImportUnits::len_time_sq_per_mass:
            return len * ipow<2>(time) / mass_scale(sys);
        case ImportUnits::size_:
            CELER_ASSERT_UNREACHABLE();
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Get the native value from a unit CLHEP quantity of this type.
 *
 * Multiply a Geant4 quantity by the result to convert to the native unit
 * system.
 */
double native_value_from_clhep(ImportUnits q)
{
    return native_value_from(celeritas::UnitSystem::clhep, q);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
