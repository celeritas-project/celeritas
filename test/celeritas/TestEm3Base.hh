//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TestEm3Base.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

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
 *
 * \todo Refactor to allow more generic geant4 problem setup, move similar code
 * with LDemo and Acceleritas into main library.
 */
class TestEm3Base : virtual public GlobalGeoTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using real_type      = celeritas::real_type;
    using ImportData     = celeritas::ImportData;
    using PhysicsOptions = celeritas::PhysicsParamsOptions;
    //!@}

  public:
    //!@{
    //! Whether the Geant4 configuration match a certain machine
    static bool is_ci_build();
    static bool is_wildstyle_build();
    static bool is_srj_build();
    //!@}

  protected:
    const char* geometry_basename() const override { return "testem3-flat"; }

    virtual bool      enable_fluctuation() const { return true; }
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
//! Print the current configuration
struct PrintableBuildConf
{
};
std::ostream& operator<<(std::ostream& os, const PrintableBuildConf&);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
