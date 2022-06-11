//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TestEm3Base.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"

#include "GlobalGeoTestBase.hh"

namespace celeritas
{
struct ImportData;
struct PhysicsParamsOptions;
} // namespace celeritas

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for replicating the AdePT TestEm3 input.
 *
 * This class requires Geant4 to import the data.
 */
class TestEm3Base : virtual public GlobalGeoTestBase
{
  public:
    //!@{
    //! Type aliases
    using real_type      = celeritas::real_type;
    using ImportData     = celeritas::ImportData;
    using PhysicsOptions = celeritas::PhysicsParamsOptions;
    //!@}

  protected:
    const char* geometry_basename() const override { return "testem3-flat"; }

    virtual bool      enable_msc() const { return false; }
    virtual real_type secondary_stack_factor() const { return 3.0; }

    SPConstMaterial    build_material() override;
    SPConstGeoMaterial build_geomaterial() override;
    SPConstParticle    build_particle() override;
    SPConstCutoff      build_cutoff() override;
    SPConstPhysics     build_physics() override;

    virtual PhysicsOptions build_physics_options() const;

    // Access lazily loaded static geant4 data
    const celeritas::ImportData& imported_data() const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
