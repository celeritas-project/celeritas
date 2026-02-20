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
#include "corecel/math/Quantity.hh"

namespace celeritas
{
//! \todo Update with #2223
//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Convert via a quantity to native LArSoft types/units
template<class Q>
double convert_to_larsoft(real_type v)
{
    return value_as<Q>(native_value_to<Q>(v));
}

//---------------------------------------------------------------------------//
//! Convert via a quantity from native LArSoft types/units
template<class Q>
real_type convert_from_larsoft(double v)
{
    return native_value_from(Q(v));
}

//---------------------------------------------------------------------------//
//! Convert via a quantity to native LArSoft types/units
template<class Q, class T>
geo::Point_t convert_to_larsoft(Array<T, 3> const& v)
{
    return {value_as<Q>(native_value_to<Q>(v[0])),
            value_as<Q>(native_value_to<Q>(v[1])),
            value_as<Q>(native_value_to<Q>(v[2]))};
}

//---------------------------------------------------------------------------//
//! Convert via a quantity from native LArSoft types/units
template<class Q>
Array<typename Q::value_type, 3> convert_from_larsoft(geo::Point_t const& v)
{
    return {native_value_from(Q(v.X())),
            native_value_from(Q(v.Y())),
            native_value_from(Q(v.Z()))};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
