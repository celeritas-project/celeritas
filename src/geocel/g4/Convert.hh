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
//! Convert a scalar via a quantity to native Geant4 types/units
template<class Q>
double native_to_geant(real_type v)
{
    return value_as<Q>(native_value_to<Q>(v));
}

//---------------------------------------------------------------------------//
/*!
 * Convert a scalar via a quantity from native Geant4 types/units.
 *
 * This performs a simultaneous unit conversion (from Geant4 to Celeritas) and
 * optional type conversion (usually from double to real_type).
 *
 * The first argument (required) is the quantity represented by the Geant4
 * value, and the second is to allow casting to \c real_type .
 *
 * \p Examples
 *
 * The track position is in native Celeritas units and value types:
 * \code
    track.position = native_from_geant<lengthunits::ClhepLength, real_type>(
      g4track.GetPosition());
   \endcode
 */
template<class Q, class T = typename Q::value_type>
T native_from_geant(double v)
{
    return static_cast<T>(native_value_from(Q(v)));
}

//---------------------------------------------------------------------------//
//! Convert an array via a quantity to native Geant4 types/units
template<class Q, class T>
G4ThreeVector native_to_geant(Array<T, 3> const& v)
{
    return to_g4vector(value_as<Q>(native_value_to<Q>(v)));
}

//---------------------------------------------------------------------------//
/*!
 * Convert via a quantity from native Geant4 types/units.
 *
 * The first argument (required) is the quantity represented by the Geant4
 * value, and the second is to allow casting to \c real_type .
 */
template<class Q, class T = typename Q::value_type>
Array<T, 3> native_from_geant(G4ThreeVector const& v)
{
    return static_array_cast<T>(
        native_value_from(make_quantity_array<Q>(to_array(v))));
}

//---------------------------------------------------------------------------//
/*!
 * Convert a C vector via a quantity from native Geant4 types/units.
 */
template<class Q, class T = typename Q::value_type>
Array<T, 3> native_from_geant(Array<double, 3> const& v)
{
    return static_array_cast<T>(native_value_from(make_quantity_array<Q>(v)));
}

//---------------------------------------------------------------------------//
// DEPRECATED
//---------------------------------------------------------------------------//
//! Value of a unit Celeritas length in the CLHEP unit system
inline constexpr double clhep_length{1 / lengthunits::millimeter};

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
