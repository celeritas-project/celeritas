//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/OnlyGeoTestBase.hh
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
 * Mixin class providing "unreachable" implementations for param construction.
 */
class OnlyGeoTestBase : virtual public GlobalTestBase
{
  public:
    SPConstParticle build_particle() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstCutoff build_cutoff() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstPhysics build_physics() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstSim build_sim() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstMaterial build_material() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstGeoMaterial build_geomaterial() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    SPConstWentzelOKVI build_wentzel() override { CELER_ASSERT_UNREACHABLE(); }
};
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
