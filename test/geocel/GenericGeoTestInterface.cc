//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestInterface.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestInterface.hh"

#include <gtest/gtest.h>

#include "corecel/Config.hh"

#include "corecel/math/SoftEqual.hh"
#include "geocel/GeantGeoUtils.hh"

#if CELERITAS_USE_GEANT4
#    include <G4VPhysicalVolume.hh>

#    include "geocel/GeantGeoParams.hh"
#endif
#include "GenericGeoResults.hh"

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
/*!
 * Get the safety tolerance (defaults to SoftEq tol).
 */
GenericGeoTrackingTolerance GenericGeoTestInterface::tracking_tol() const
{
    GenericGeoTrackingTolerance result;
    result.distance = SoftEqual{}.rel();
    result.normal = celeritas::sqrt_tol();
    result.safety = result.distance;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Whether surface normals work for the current geometry/test.
 *
 * This defaults to true and should be disabled per geometry
 * implementation/geometry class.
 */
bool GenericGeoTestInterface::supports_surface_normal() const
{
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Get the threshold for a movement being a "bump".
 *
 * This unitless tolerance is multiplied by the test's unit length when used.
 */
real_type GenericGeoTestInterface::bump_tol() const
{
    return 1e-7;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
