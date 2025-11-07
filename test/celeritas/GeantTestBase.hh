//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

  protected:
    virtual GeantPhysicsOptions build_geant_options() const;

    SPConstTrackInit build_init() override;
    SPConstAction build_along_step() override;
    SPConstGeantGeo build_geant_geo(std::string const&) const override;

    // Access lazily loaded static geant4 data
    ImportData const& imported_data() const final;

    // Import data potentially with different selection options
    virtual GeantImportDataSelection build_import_data_selection() const;

  private:
    struct ImportSetup;

    // Lazily load static geant4 data
    ImportSetup const& load(std::string const& filename = {}) const;
};

//---------------------------------------------------------------------------//
//! Print the current configuration
struct StreamableBuildConf
{
};
std::ostream& operator<<(std::ostream& os, StreamableBuildConf const&);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
