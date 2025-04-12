//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GeantTestBase.cc
//---------------------------------------------------------------------------//
#include "GeantTestBase.hh"

#include <string>

#include "corecel/Config.hh"

#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "celeritas/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/track/TrackInitParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct GeantTestBase::ImportHelper
{
    // NOTE: the import function must be static for now so that Vecgeom or
    // other clients can access Geant4 after importing the data.
    std::unique_ptr<GeantImporter> import;
    std::string geometry_basename{};
    GeantPhysicsOptions options{};
    GeantImportDataSelection selection{};
    std::unique_ptr<ScopedGeantExceptionHandler> scoped_exceptions;
    ImportData imported;
};

class GeantTestBase::CleanupGeantEnvironment : public ::testing::Environment
{
  public:
    void SetUp() override {}
    void TearDown() override
    {
        ImportHelper& i = GeantTestBase::import_helper();
        i = {};
    }
};

//---------------------------------------------------------------------------//
//! Whether results should be equivalent to the CI build
bool GeantTestBase::is_ci_build()
{
    return CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
           && CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_GEANT4
           && CELERITAS_UNITS == CELERITAS_UNITS_CGS
           && cstring_equal(cmake::core_rng, "xorwow")
           && (cstring_equal(cmake::clhep_version, "2.4.6.0")
               || cstring_equal(cmake::clhep_version, "2.4.6.4")
               || cstring_equal(cmake::clhep_version, "2.4.7.1"))
           && cstring_equal(cmake::geant4_version, "11.3.0");
}

//---------------------------------------------------------------------------//
//! Whether Geant4 dependencies match those on Wildstyle
bool GeantTestBase::is_wildstyle_build()
{
    return GeantTestBase::is_ci_build();
}

//---------------------------------------------------------------------------//
//! Whether Geant4 dependencies match those on Summit
bool GeantTestBase::is_summit_build()
{
    return GeantTestBase::is_ci_build();
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
auto GeantTestBase::build_geant_options() const -> GeantPhysicsOptions
{
    GeantPhysicsOptions options;
    options.em_bins_per_decade = 14;
    options.rayleigh_scattering = false;
    return options;
}

//---------------------------------------------------------------------------//
auto GeantTestBase::build_init() -> SPConstTrackInit
{
    TrackInitParams::Input input;
    input.capacity = 4096 * 2;
    input.max_events = 4096;
    input.track_order = TrackOrder::none;
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
auto GeantTestBase::build_fresh_geometry(std::string_view filename) -> SPConstGeoI
{
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE \
    && CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_DOUBLE
    // Load fake version of geometry because Geant4 conversion isn't available
    return Base::build_fresh_geometry(filename);
#else
    // Import geometry directly from in-memory Geant4
    CELER_LOG(info) << "Importing geometry from Geant4 (instead of directly "
                       "from "
                    << filename << ")";
    auto* world = this->get_world_volume();
    CELER_EXPECT(world);
    return std::make_shared<GeoParams>(world);
#endif
}

//---------------------------------------------------------------------------//
// Lazily set up and load geant4
auto GeantTestBase::imported_data() const -> ImportData const&
{
    GeantPhysicsOptions opts = this->build_geant_options();
    GeantImportDataSelection sel = this->build_import_data_selection();

    ImportHelper& i = GeantTestBase::import_helper();
    if (!i.import)
    {
        i.geometry_basename = this->geometry_basename();
        i.options = opts;
        std::string gdml_inp = this->test_data_path(
            "geocel", (i.geometry_basename + ".gdml").c_str());
        i.import = std::make_unique<GeantImporter>(
            GeantSetup{gdml_inp.c_str(), i.options});
        i.scoped_exceptions = std::make_unique<ScopedGeantExceptionHandler>();
        i.imported = (*i.import)(sel);
        i.options.verbose = false;
        i.selection = sel;
    }
    else
    {
        // Verbosity change is allowable
        opts.verbose = false;

        static char const explanation[]
            = R"( (Geant4 cannot be set up twice in one execution: see issue #462))";
        CELER_VALIDATE(this->geometry_basename() == i.geometry_basename,
                       << "cannot load new geometry '"
                       << this->geometry_basename() << "' when another '"
                       << i.geometry_basename << "' was already set up"
                       << explanation);
        CELER_VALIDATE(opts == i.options,
                       << "cannot change physics options after setup "
                       << explanation);

        if (sel != i.selection)
        {
            // Reload with new selection
            i.imported = (*i.import)(sel);
            i.selection = sel;
        }
    }
    return i.imported;
}

//---------------------------------------------------------------------------//
GeantImportDataSelection GeantTestBase::build_import_data_selection() const
{
    // By default, don't try to import optical data
    GeantImportDataSelection result;
    result.processes &= (~GeantImportDataSelection::optical);
    return result;
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
        CELER_LOG(debug) << "Registering CleanupGeantEnvironment";
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
    os << "RNG=\"" << cmake::core_rng << "\", CLHEP=\"" << cmake::clhep_version
       << "\", Geant4=\"" << cmake::geant4_version << '"';
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
