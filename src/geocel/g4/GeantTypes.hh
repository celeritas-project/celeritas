//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantTypes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <G4ThreeVector.hh>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Convert a Geant4 vector to a Celeritas array with full precision
inline Array<double, 3> to_array(G4ThreeVector const& v)
{
    return {v.x(), v.y(), v.z()};
}

//---------------------------------------------------------------------------//
//! Convert a native Celeritas 3-vector to a Geant4 equivalent
inline G4ThreeVector to_g4vector(Array<double, 3> const& arr)
{
    return {arr[0], arr[1], arr[2]};
}

//! \cond (CELERITAS_DOC_DEV)
//---------------------------------------------------------------------------//
//! Let y <- a * x + y
inline void axpy(double a, G4ThreeVector const& x, G4ThreeVector* y)
{
    CELER_EXPECT(y);
    for (int i = 0; i < 3; ++i)
    {
        (*y)[i] = std::fma(a, x[i], (*y)[i]);
    }
}
//! \endcond

//---------------------------------------------------------------------------//
}  // namespace celeritas
