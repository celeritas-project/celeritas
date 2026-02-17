//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/UnitUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "geocel/detail/LengthUnits.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Whether the current geometry can correctly scale the input as needed
inline constexpr bool unit_scaling_enabled
    = (CELERITAS_UNITS == CELERITAS_UNITS_CGS
       || CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE
       || CELERITAS_USE_GEANT4);

//---------------------------------------------------------------------------//
//! Convert a value to centimeters from the native system
CELER_CONSTEXPR_FUNCTION real_type to_cm(real_type v)
{
    return v / ::celeritas::lengthunits::centimeter;
}

//---------------------------------------------------------------------------//
//! Convert a value *from* centimeters to the native system
CELER_CONSTEXPR_FUNCTION real_type from_cm(real_type v)
{
    return v * ::celeritas::lengthunits::centimeter;
}

//---------------------------------------------------------------------------//
//! Convert an array to centimeters from the native system
CELER_CONSTEXPR_FUNCTION Array<real_type, 3> to_cm(Array<real_type, 3> const& v)
{
    return {to_cm(v[0]), to_cm(v[1]), to_cm(v[2])};
}

//---------------------------------------------------------------------------//
//! Convert an array *from* centimeters to the native system
CELER_CONSTEXPR_FUNCTION Array<real_type, 3>
from_cm(Array<real_type, 3> const& v)
{
    return {from_cm(v[0]), from_cm(v[1]), from_cm(v[2])};
}

//---------------------------------------------------------------------------//
//! Unit system used for reference results in a test
struct UnitLength
{
    Constant value{::celeritas::lengthunits::centimeter};
    std::string label{"cm"};

    template<class T>
    constexpr T from_native(T const& v) const
    {
        return v / value;
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
