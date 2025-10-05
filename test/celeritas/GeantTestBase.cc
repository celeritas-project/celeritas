//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GeantTestBase.cc
//---------------------------------------------------------------------------//
#include "GeantTestBase.hh"

#include <memory>
#include <string>

#include "corecel/Config.hh"

#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Version.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "celeritas/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/geo/CoreGeoTraits.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/setup/Import.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "PersistentSP.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Keep Geant4 setup persistently across tests
struct GeantTestBase::ImportSetup
{
    std::unique_ptr<GeantImporter> import;
    std::shared_ptr<GeantGeoParams> geo;
    GeantPhysicsOptions options{};
    GeantImportDataSelection selection{};
    ImportData imported;
    ScopedGeantExceptionHandler scoped_exceptions;
};

//---------------------------------------------------------------------------//
//! Whether results should be equivalent to the main CI build
bool GeantTestBase::is_ci_build()
{
    if (!(CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
          && CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_GEANT4
          && CELERITAS_UNITS == CELERITAS_UNITS_CGS)
        && cstring_equal(cmake::core_rng, "xorwow"))
    {
        // Config options are different
        return false;
    }
    // Check clhep/g4 versions
    auto clhep = Version::from_string(cmake::clhep_version);
    auto g4 = Version::from_string(cmake::geant4_version);
    return clhep >= Version{2, 4, 6} && clhep < Version{2, 5}
           && g4 >= Version{11, 3} && g4 < Version{11, 4};
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
auto GeantTestBase::build_geant_geo(std::string const& filename) const
    -> SPConstGeantGeo
{
    CELER_LOG(debug) << "Accessing Geant4 geometry by setting up problem";

    ImportSetup const& i = this->load(filename);
    CELER_ENSURE(i.geo);
    return i.geo;
}

//---------------------------------------------------------------------------//
// Lazily set up and load geant4
auto GeantTestBase::imported_data() const -> ImportData const&
{
    this->lazy_geo();
    return this->load({}).imported;
}

auto GeantTestBase::load(std::string const& filename) const
    -> ImportSetup const&
{
    GeantPhysicsOptions opts = this->build_geant_options();
    GeantImportDataSelection sel = this->build_import_data_selection();

    using PersistentImportSetup = PersistentSP<GeantTestBase::ImportSetup>;

    static PersistentImportSetup ps{"Geant4 import"};
    std::shared_ptr<ImportSetup> i;
    bool stale{true};
    if (!ps)
    {
        CELER_VALIDATE(!filename.empty(),
                       << "load was called before build_geant_geo");
        i = std::make_shared<ImportSetup>();
        i->options = opts;
        i->import = std::make_unique<GeantImporter>(GeantSetup{filename, opts});
        i->geo = i->import->geo_params();
        CELER_ASSERT(i->geo);
        CELER_ASSERT(!celeritas::global_geant_geo().expired());

        ps.set(filename, i);
    }
    else
    {
        // Verbosity change is allowable
        opts.verbose = false;

        static char const explanation[]
            = R"( (Geant4 cannot be set up twice in one execution: see issue #462))";
        CELER_VALIDATE(filename.empty() || filename == ps.key(),
                       << "cannot load new problem '" << filename
                       << "' when another '" << ps.key()
                       << "' was already set up" << explanation);
        i = ps.value();
        CELER_ASSERT(i);
        CELER_VALIDATE(opts == i->options,
                       << "cannot change physics options after setup "
                       << explanation);

        if (sel == i->selection)
        {
            // No need to reload
            stale = false;
        }
    }

    if (stale)
    {
        CELER_ASSERT(i->import);
        i->imported = (*i->import)(sel);
        i->selection = sel;
        i->options.verbose = false;

        if (i->selection.reader_data)
        {
            setup::physics_from(inp::PhysicsFromGeantFiles{}, i->imported);
        }
    }

    CELER_ENSURE(i);
    return *i;
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
std::ostream& operator<<(std::ostream& os, StreamableBuildConf const&)
{
    os << "RNG=\"" << cmake::core_rng << "\", CLHEP=\"" << cmake::clhep_version
       << "\", Geant4=\"" << cmake::geant4_version << '"';
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
