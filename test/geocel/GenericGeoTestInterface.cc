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

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
/*!
 * Get the safety tolerance (defaults to SoftEq tol).
 */
real_type GenericGeoTestInterface::safety_tol() const
{
    constexpr SoftEqual<> default_seq{};
    return default_seq.rel();
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
