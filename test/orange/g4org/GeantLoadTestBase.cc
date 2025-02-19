//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/GeantLoadTestBase.cc
//---------------------------------------------------------------------------//
#include "GeantLoadTestBase.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGdmlLoader.hh"
#include "geocel/GeantGeoUtils.hh"

namespace celeritas
{
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//
std::string GeantLoadTestBase::loaded_filename_{};
G4VPhysicalVolume* GeantLoadTestBase::world_volume_{nullptr};

//---------------------------------------------------------------------------//
/*!
 * Helper function: build via Geant4 GDML reader.
 */
G4VPhysicalVolume const*
GeantLoadTestBase::load_gdml(std::string const& filename)
{
    CELER_EXPECT(!filename.empty());
    if (filename == loaded_filename_)
    {
        return world_volume_;
    }

    if (world_volume_)
    {
        // Clear old geant4 data
        TearDownTestSuite();
    }
    ::celeritas::test::ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                                   LogLevel::warning};
    world_volume_ = ::celeritas::load_gdml(filename);
    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
    loaded_filename_ = filename;

    return world_volume_;
}

//---------------------------------------------------------------------------//
/*!
 * Load a test input.
 */
G4VPhysicalVolume const*
GeantLoadTestBase::load_test_gdml(std::string_view basename)
{
    return this->load_gdml(
        this->test_data_path("geocel", std::string(basename) + ".gdml"));
}

//---------------------------------------------------------------------------//
/*!
 * Reset the geometry.
 */
void GeantLoadTestBase::TearDownTestSuite()
{
    loaded_filename_ = {};
    ::celeritas::reset_geant_geometry();
    world_volume_ = nullptr;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
