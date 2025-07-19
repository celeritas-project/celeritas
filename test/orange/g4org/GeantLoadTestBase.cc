//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/GeantLoadTestBase.cc
//---------------------------------------------------------------------------//
#include "GeantLoadTestBase.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "geocel/GeantGeoParams.hh"

namespace celeritas
{
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Helper function: build via Geant4 GDML reader.
 */
void GeantLoadTestBase::load_gdml(std::string const& filename)
{
    auto geo = this->get_geometry(filename);
    geo_ = std::dynamic_pointer_cast<GeantGeoParams const>(geo);
    CELER_ENSURE(geo_);
}

//---------------------------------------------------------------------------//
/*!
 * Load a test input.
 */
void GeantLoadTestBase::load_test_gdml(std::string_view basename)
{
    return this->load_gdml(std::string{"test:"} + std::string{basename});
}

//---------------------------------------------------------------------------//
/*!
 * Access the geo params after loading.
 */
GeantGeoParams const& GeantLoadTestBase::geo() const
{
    CELER_VALIDATE(geo_, << "geometry was not loaded into the test harness");
    return *geo_;
}

//---------------------------------------------------------------------------//
/*!
 * Access the geo params after loading.
 */
G4VPhysicalVolume const& GeantLoadTestBase::world() const
{
    return *this->geo().world();
}

//---------------------------------------------------------------------------//
/*!
 * Construct a fresh geometry from a filename
 */
auto GeantLoadTestBase::build_fresh_geometry(std::string_view key)
    -> SPConstGeoI
{
    std::string filename{key};
    if (starts_with(filename, "test:"))
    {
        filename = this->test_data_path("geocel", filename.substr(5) + ".gdml");
    }

    SPConstGeoI result;
    {
        ::celeritas::test::ScopedLogStorer scoped_log_{
            &celeritas::self_logger(), LogLevel::warning};
        result = GeantGeoParams::from_gdml(filename);
        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
