//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/SimpleTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "GlobalTestBase.hh"
#include "OnlyCoreTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Compton scattering with gammas in mock aluminum in a box in hard vacuum.
 */
class SimpleTestBase : virtual public GlobalTestBase, public OnlyCoreTestBase
{
  protected:
    std::string_view gdml_basename() const override { return "two-boxes"; }

    virtual real_type secondary_stack_factor() const { return 1.0; }

    SPConstMaterial build_material() override;
    SPConstGeoMaterial build_geomaterial() override;
    SPConstParticle build_particle() override;
    SPConstCutoff build_cutoff() override;
    SPConstPhysics build_physics() override;
    SPConstSim build_sim() override;
    SPConstTrackInit build_init() override;
    SPConstAction build_along_step() override;
    SPConstWentzelOKVI build_wentzel() override;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
