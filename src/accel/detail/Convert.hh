//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/Convert.hh
//---------------------------------------------------------------------------//
#pragma once

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4ThreeVector.hh>

#include "corecel/Types.hh"
#include "orange/Types.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
inline T convert_from_geant(T const& val, T units)
{
    return val / units;
}

//---------------------------------------------------------------------------//
inline Real3 convert_from_geant(G4ThreeVector const& vec, double units)
{
    return {vec[0] / units, vec[1] / units, vec[2] / units};
}

//---------------------------------------------------------------------------//
template<class T>
inline T convert_to_geant(T const& val, T units)
{
    return val * units;
}

//---------------------------------------------------------------------------//
inline G4ThreeVector convert_to_geant(Real3 const& arr, double units)
{
    return {arr[0] * units, arr[1] * units, arr[2] * units};
}

//---------------------------------------------------------------------------//
inline double convert_to_geant(units::MevEnergy const& energy, double units)
{
    CELER_EXPECT(units == CLHEP::MeV);
    return energy.value() * CLHEP::MeV;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
