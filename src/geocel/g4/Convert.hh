//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/Convert.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayQuantity.hh"
#include "geocel/detail/LengthQuantities.hh"

#include "GeantTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Convert via a quantity to native Geant4 types/units
template<class Q, class T>
G4ThreeVector convert_to_geant(Array<T, 3> const& v)
{
    return to_g4vector(value_as<Q>(native_value_to<Q>(v)));
}

//---------------------------------------------------------------------------//
//! Convert via a quantity to native Geant4 types/units
template<class Q, class T>
Array<typename Q::value_type, 3> convert_from_geant(G4ThreeVector const& v)
{
    return native_value_from<Q>(make_quantity_array<Q>(to_array(v)));
}

//---------------------------------------------------------------------------//
//! Convert via a quantity to native Geant4 types/units
template<class Q>
double convert_to_geant(real_type v)
{
    return value_as<Q>(native_value_to<Q>(v));
}

//---------------------------------------------------------------------------//
//! Convert via a quantity to native Geant4 types/units
template<class Q>
real_type convert_from_geant(double v)
{
    return native_value_from<Q>(Q(v));
}

//---------------------------------------------------------------------------//
// DEPRECATED
//---------------------------------------------------------------------------//
/*!
 * Convert a value from Geant4 with CLHEP units.
 */
constexpr inline double convert_from_geant(real_type val, double units)
{
    return val / units;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a 3-vector from Geant4/CLHEP to Celeritas native units.
 */
inline Real3 convert_from_geant(G4ThreeVector const& vec, double units)
{
    return static_array_cast<real_type>(to_array(vec) / units);
}

//---------------------------------------------------------------------------//
/*!
 * Convert a C array from Geant4/CLHEP to Celeritas native units.
 */
inline Real3 convert_from_geant(double const vec[3], double units)
{
    return static_array_cast<real_type>(Array{vec[0], vec[1], vec[2]} / units);
}

//---------------------------------------------------------------------------//
/*!
 * Convert a native Celeritas quantity to a Geant4 value with CLHEP units.
 */
constexpr inline double convert_to_geant(real_type val, double units)
{
    return val * units;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a native Celeritas 3-vector to a Geant4 equivalent.
 */
inline G4ThreeVector convert_to_geant(Real3 const& arr, double units)
{
    return to_g4vector(static_array_cast<double>(arr) * units);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
