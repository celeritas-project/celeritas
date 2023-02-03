//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GeantTestBase.cc
//---------------------------------------------------------------------------//
#include "GeantTestBase.hh"

#include <string>

#include "celeritas_cmake_strings.h"
#include "corecel/io/Logger.hh"
#include "celeritas/em/msc/UrbanMscParams.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/track/TrackInitParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
namespace
{
//---------------------------------------------------------------------------//
//! Test for equality of two C strings
bool cstring_equal(char const* lhs, char const* rhs)
{
    return std::strcmp(lhs, rhs) == 0;
}

//---------------------------------------------------------------------------//
}  // namespace

struct GeantTestBase::ImportHelper
{
    // NOTE: the import function must be static for now so that Vecgeom or
    // other clients can access Geant4 after importing the data.
    std::unique_ptr<GeantImporter> import;
    std::string geometry_basename{};
    GeantPhysicsOptions options{};
    ImportData imported;
};

class GeantTestBase::CleanupGeantEnvironment : public ::testing::Environment
{
  public:
    void SetUp() override {}
    void TearDown() override
    {
        ImportHelper& i = GeantTestBase::import_helper();
        CELER_LOG(debug) << "Destroying '" << i.geometry_basename
                         << "' Geant4 run manager";
        i = {};
    }
};

//---------------------------------------------------------------------------//
//! Whether Geant4 dependencies match those on the CI build
bool GeantTestBase::is_ci_build()
{
    return cstring_equal(celeritas_rng, "XORWOW")
           && cstring_equal(celeritas_clhep_version, "2.4.6.0")
           && cstring_equal(celeritas_geant4_version, "11.0.3");
}

//---------------------------------------------------------------------------//
//! Whether Geant4 dependencies match those on Wildstyle
bool GeantTestBase::is_wildstyle_build()
{
    return cstring_equal(celeritas_rng, "XORWOW")
           && cstring_equal(celeritas_clhep_version, "2.4.6.0")
           && cstring_equal(celeritas_geant4_version, "11.0.3");
}

//---------------------------------------------------------------------------//
//! Whether Geant4 dependencies match those on Summit
bool GeantTestBase::is_summit_build()
{
    return cstring_equal(celeritas_rng, "XORWOW")
           && cstring_equal(celeritas_clhep_version, "2.4.5.1")
           && cstring_equal(celeritas_geant4_version, "11.0.0");
}

//---------------------------------------------------------------------------//
//! Get the Geant4 top-level geometry element (immutable)
G4VPhysicalVolume const* GeantTestBase::get_world_volume() const
{
    return GeantImporter::get_world_volume();
}

//---------------------------------------------------------------------------//
//! Get the Geant4 top-level geometry element
G4VPhysicalVolume const* GeantTestBase::get_world_volume()
{
    // Load geometry
    this->imported_data();
    return const_cast<GeantTestBase const*>(this)->get_world_volume();
}

//---------------------------------------------------------------------------//
// PROTECTED MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
auto GeantTestBase::build_init() -> SPConstTrackInit
{
    TrackInitParams::Input input;
    input.capacity = 4096;
    input.max_events = 4096;
    return std::make_shared<TrackInitParams>(input);
}

//---------------------------------------------------------------------------//
auto GeantTestBase::build_along_step() -> SPConstAction
{
    auto& action_reg = *this->action_reg();
    auto msc = UrbanMscParams::from_import(
        *this->particle(), *this->material(), this->imported_data());
    auto result = AlongStepGeneralLinearAction::from_params(
        action_reg.next_id(),
        *this->material(),
        *this->particle(),
        msc,
        this->imported_data().em_params.energy_loss_fluct);
    CELER_ASSERT(result);
    CELER_ASSERT(result->has_fluct()
                 == this->build_geant_options().eloss_fluctuation);
    CELER_ASSERT(
        result->has_msc()
        == (this->build_geant_options().msc != MscModelSelection::none));
    action_reg.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
auto GeantTestBase::build_geant_options() const -> GeantPhysicsOptions
{
    GeantPhysicsOptions options;
    options.em_bins_per_decade = 14;
    options.rayleigh_scattering = false;
    return options;
}

//---------------------------------------------------------------------------//
// Lazily set up and load geant4
auto GeantTestBase::imported_data() const -> ImportData const&
{
    ImportHelper& i = GeantTestBase::import_helper();
    if (!i.import)
    {
        i.geometry_basename = this->geometry_basename();
        i.options = this->build_geant_options();
        std::string gdml_inp = this->test_data_path(
            "celeritas", (i.geometry_basename + ".gdml").c_str());
        i.import = std::make_unique<GeantImporter>(
            GeantSetup{gdml_inp.c_str(), i.options});
        i.imported = (*i.import)();
    }
    else
    {
        static char const explanation[]
            = " (Geant4 cannot be set up twice in one execution: see issue "
              "#462)";
        CELER_VALIDATE(this->geometry_basename() == i.geometry_basename,
                       << "cannot load new geometry '"
                       << this->geometry_basename() << "' when another '"
                       << i.geometry_basename << "' was already set up"
                       << explanation);
        CELER_VALIDATE(this->build_geant_options() == i.options,
                       << "cannot change physics options after "
                       << explanation);
    }
    CELER_ENSURE(i.imported);
    return i.imported;
}

//---------------------------------------------------------------------------//
auto GeantTestBase::import_helper() -> ImportHelper&
{
    static bool registered_cleanup = false;
    if (!registered_cleanup)
    {
        /*! Always reset Geant4 at end of testing before global destructors.
         *
         * This is needed because Geant4 is filled with static data, so we must
         * destroy our references before it gets cleaned up.
         */
        CELER_LOG(debug) << "Registering CleanupGeoEnvironment";
        ::testing::AddGlobalTestEnvironment(new CleanupGeantEnvironment());
        registered_cleanup = true;
    }

    // Delayed initialization
    static ImportHelper i;
    return i;
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, PrintableBuildConf const&)
{
    os << "RNG=\"" << celeritas_rng << "\", CLHEP=\""
       << celeritas_clhep_version << "\", Geant4=\""
       << celeritas_geant4_version << '"';
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
