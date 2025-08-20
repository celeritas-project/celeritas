//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/OnlyCoreTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"

#include "GlobalTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Mixin class providing "unreachable" implementations for optical params
 * construction.
 */
class OnlyCoreTestBase : virtual public GlobalTestBase
{
  public:
    SPConstCherenkov build_cherenkov() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstOpticalMaterial build_optical_material() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    SPConstOpticalPhysics build_optical_physics() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    SPConstOpticalSurfacePhysics build_optical_surface_physics() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    SPConstScintillation build_scintillation() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
};
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
