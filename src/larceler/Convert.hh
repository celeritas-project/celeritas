//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/Convert.hh
//---------------------------------------------------------------------------//
#pragma once

#include <larcoreobj/SimpleTypesAndConstants/geo_vectors.h>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/ArrayQuantity.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/UnitTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

using LarsoftTime = Quantity<celeritas::units::Nanosecond, double>;
using LarsoftLen = Quantity<celeritas::units::Centimeter, double>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Convert via a quantity to native LArSoft types/units
template<class Q>
inline double convert_to_larsoft(real_type v)
{
    return value_as<Q>(native_value_to<Q>(v));
}

//---------------------------------------------------------------------------//
//! Convert via a quantity from native LArSoft types/units
template<class Q>
inline real_type convert_from_larsoft(double v)
{
    return native_value_from(Q(v));
}

//---------------------------------------------------------------------------//
//! Convert from a Celeritas array to a ROOT point
inline geo::Point_t to_larpoint(Array<double, 3> const& arr)
{
    return {arr[0], arr[1], arr[2]};
}

//---------------------------------------------------------------------------//
//! Convert a ROOT point to a Celeritas array
inline Array<double, 3> to_array(geo::Point_t const& v)
{
    return {v.X(), v.Y(), v.Z()};
}

//---------------------------------------------------------------------------//
//! Convert via a quantity to native LArSoft types/units
template<class Q>
geo::Point_t convert_to_larsoft(Array<double, 3> const& v)
{
    return to_larpoint(value_as<Q>(native_value_to<Q>(v)));
}

//---------------------------------------------------------------------------//
//! Convert via a quantity from native LArSoft types/units
template<class Q>
Array<typename Q::value_type, 3> convert_from_larsoft(geo::Point_t const& v)
{
    return native_value_from(make_quantity_array<Q>(to_array(v)));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
