//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/Convert.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4ThreeVector.hh>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Constant.hh"
#include "geocel/Types.hh"
#include "geocel/detail/LengthUnits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// CONSTANTS
//---------------------------------------------------------------------------//
//! Value of a unit Celeritas length in the CLHEP unit system
inline constexpr double clhep_length{1 / lengthunits::millimeter};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Convert a value from Geant4/CLHEP to Celeritas native units.
 */
template<class T>
constexpr inline T convert_from_geant(T const& val, T units)
{
    return val / units;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a value from Geant4 with CLHEP units.
 */
constexpr inline double convert_from_geant(double val, double units)
{
    return val / units;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a 3-vector from Geant4/CLHEP to Celeritas native units.
 */
inline Real3 convert_from_geant(G4ThreeVector const& vec, double units)
{
    return {static_cast<real_type>(vec[0] / units),
            static_cast<real_type>(vec[1] / units),
            static_cast<real_type>(vec[2] / units)};
}

//---------------------------------------------------------------------------//
/*!
 * Convert a C array from Geant4/CLHEP to Celeritas native units.
 */
inline Real3 convert_from_geant(double const vec[3], double units)
{
    return {static_cast<real_type>(vec[0] / units),
            static_cast<real_type>(vec[1] / units),
            static_cast<real_type>(vec[2] / units)};
}

//---------------------------------------------------------------------------//
/*!
 * Convert a native Celeritas quantity to a Geant4 value with CLHEP units.
 */
template<class T>
constexpr inline double convert_to_geant(T const& val, double units)
{
    return val * units;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a native Celeritas quantity to a Geant4 value.
 */
constexpr inline double convert_to_geant(real_type val, double units)
{
    return double{val} * units;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a native Celeritas 3-vector to a Geant4 equivalent.
 */
template<class T>
inline G4ThreeVector convert_to_geant(Array<T, 3> const& arr, double units)
{
    return {arr[0] * units, arr[1] * units, arr[2] * units};
}

//! \cond (CELERITAS_DOC_DEV)
//---------------------------------------------------------------------------//
/*!
 * Set y += a * x .
 */
inline void axpy(double a, G4ThreeVector const& x, G4ThreeVector* y)
{
    CELER_EXPECT(y);
    for (int i = 0; i < 3; ++i)
    {
        (*y)[i] = a * x[i] + (*y)[i];
    }
}
//! \endcond

//---------------------------------------------------------------------------//
}  // namespace celeritas
