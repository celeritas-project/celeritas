//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GeantTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

#include "celeritas/Types.hh"

#include "ImportedDataTestBase.hh"

class G4VPhysicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
struct GeantPhysicsOptions;
struct GeantImportDataSelection;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for loading problem data through Geant4.
 */
class GeantTestBase : public ImportedDataTestBase
{
    using Base = ImportedDataTestBase;

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
    virtual GeantPhysicsOptions build_geant_options() const;

    SPConstTrackInit build_init() override;
    SPConstAction build_along_step() override;
    SPConstGeoI build_fresh_geometry(std::string_view) override;

    // Access lazily loaded static geant4 data
    ImportData const& imported_data() const final;

    // Import data potentially with different selection options
    virtual GeantImportDataSelection build_import_data_selection() const;

  private:
    struct ImportHelper;
    class CleanupGeantEnvironment;

    static ImportHelper& import_helper();
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
