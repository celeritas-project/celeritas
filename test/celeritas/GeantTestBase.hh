//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GeantTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

#include "celeritas/Types.hh"

#include "GlobalGeoTestBase.hh"

class G4VPhysicalVolume;

namespace celeritas
{
struct ImportData;
struct PhysicsParamsOptions;
}  // namespace celeritas

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for simple Geant4 testem3-like problems.
 */
class GeantTestBase : virtual public GlobalGeoTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using PhysicsOptions = PhysicsParamsOptions;
    //!@}

  public:
    //!@{
    //! Whether the Geant4 configuration match a certain machine
    static bool is_ci_build();
    static bool is_wildstyle_build();
    static bool is_summit_build();
    //!@}

    //!@{
    //! Get the Geant4 top-level geometry element
    G4VPhysicalVolume const* get_world_volume();
    G4VPhysicalVolume const* get_world_volume() const;
    //!@}

  protected:
    virtual bool enable_fluctuation() const = 0;
    virtual bool enable_msc() const = 0;
    virtual bool combined_brems() const = 0;
    virtual real_type secondary_stack_factor() const = 0;

    SPConstMaterial build_material() override;
    SPConstGeoMaterial build_geomaterial() override;
    SPConstParticle build_particle() override;
    SPConstCutoff build_cutoff() override;
    SPConstPhysics build_physics() override;
    SPConstTrackInit build_init() override;
    SPConstAction build_along_step() override;

    virtual PhysicsOptions build_physics_options() const;

    // Access lazily (re)loaded static geant4 data
    ImportData const& imported_data() const;
};

//---------------------------------------------------------------------------//
//! Print the current configuration
struct PrintableBuildConf
{
};
std::ostream& operator<<(std::ostream& os, PrintableBuildConf const&);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
