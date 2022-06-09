//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GlobalTestBase.cc
//---------------------------------------------------------------------------//
#include "GlobalTestBase.hh"

#include <fstream>

#include "celeritas_config.h"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputManager.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/global/ActionManagerOutput.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/phys/PhysicsParamsOutput.hh"
#include "celeritas/random/RngParams.hh"
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// GlobalGeoTestBase
//---------------------------------------------------------------------------//
auto GlobalGeoTestBase::build_geometry() -> SPConstGeo
{
    auto& lazy = GlobalGeoTestBase::lazy_geo();

    // Construct filename
    const char* basename_cstr = this->geometry_basename();
    CELER_ASSERT(basename_cstr);
    std::string basename{basename_cstr};
    CELER_ASSERT(!basename.empty());

    if (basename != lazy.basename)
    {
        // Construct filename:
        // ${SOURCE}/test/celeritas/data/${basename}${fileext}
        const char* ext      = CELERITAS_USE_VECGEOM ? ".gdml" : ".org.json";
        auto        filename = basename + ext;
        std::string test_file
            = this->test_data_path("celeritas", filename.c_str());

        // MUST reset geometry before trying to build a new one
        // since VecGeom is all full of globals
        GlobalGeoTestBase::reset_geometry();
        lazy.geo      = std::make_shared<GeoParams>(test_file.c_str());
        lazy.basename = std::move(basename);
    }

    return lazy.geo;
}

//---------------------------------------------------------------------------//
/*!
 * Destroy the geometry if needed.
 */
void GlobalGeoTestBase::reset_geometry()
{
    auto& lazy = GlobalGeoTestBase::lazy_geo();
    if (lazy.geo)
    {
        CELER_LOG(debug) << "Destroying '" << lazy.basename << "' geometry";
        lazy.geo.reset();
    }
}

//---------------------------------------------------------------------------//
auto GlobalGeoTestBase::lazy_geo() -> LazyGeo&
{
    // Delayed initialization
    static LazyGeo lg;

    if (!lg.geo)
    {
        /*! Always reset geometry at end of testing before global destructors.
         *
         * This is needed because VecGeom stores its objects as static globals,
         * and only makes those objects visible with references/raw data. Thus
         * we can't guarantee that the GeoParams destructor is calling a valid
         * global VecGeom pointer when it destructs, since static
         * initialization/destruction order is undefined across translation
         * units.
         */
        ::testing::AddGlobalTestEnvironment(new CleanupGeoEnvironment());
    }

    return lg;
}

//---------------------------------------------------------------------------//
// GlobalTestBase
//---------------------------------------------------------------------------//
GlobalTestBase::GlobalTestBase()
{
    output_ = std::make_shared<OutputManager>();
}

//---------------------------------------------------------------------------//
// Default destructor
GlobalTestBase::~GlobalTestBase() = default;

//---------------------------------------------------------------------------//
auto GlobalTestBase::build_rng() const -> SPConstRng
{
    return std::make_shared<RngParams>(20220511);
}

//---------------------------------------------------------------------------//
auto GlobalTestBase::build_action_mgr() const -> SPActionManager
{
    ActionManager::Options opts;
    opts.sync = true;
    return std::make_shared<ActionManager>(opts);
}

//---------------------------------------------------------------------------//
auto GlobalTestBase::build_core() -> SPConstCore
{
    CoreParams::Input inp;
    inp.geometry    = this->geometry();
    inp.material    = this->material();
    inp.geomaterial = this->geomaterial();
    inp.particle    = this->particle();
    inp.cutoff      = this->cutoff();
    inp.physics     = this->physics();
    inp.rng         = this->rng();
    inp.action_mgr  = this->action_mgr();
    CELER_ASSERT(inp);
    return std::make_shared<CoreParams>(std::move(inp));
}

//---------------------------------------------------------------------------//
void GlobalTestBase::write_output()
{
    if (!CELERITAS_USE_JSON)
    {
        CELER_LOG(error) << "JSON unavailable: cannot write output";
        return;
    }
    std::string   filename = this->make_unique_filename(".json");
    std::ofstream of(filename);
    this->write_output(of);
    CELER_LOG(info) << "Wrote output to " << filename;
}

//---------------------------------------------------------------------------//
void GlobalTestBase::write_output(std::ostream& os) const
{
#if CELERITAS_USE_JSON
    JsonPimpl json_wrap;
    output_->output(&json_wrap);

    // Print with pretty indentation
    os << json_wrap.obj.dump(1) << '\n';
#else
    os << "\"output unavailable\"";
#endif
}

//---------------------------------------------------------------------------//
// PRIVATE
//---------------------------------------------------------------------------//
void GlobalTestBase::register_physics_output()
{
    CELER_ASSERT(physics_);
    output_->insert(std::make_shared<PhysicsParamsOutput>(physics_));
}

//---------------------------------------------------------------------------//
void GlobalTestBase::register_action_mgr_output()
{
    CELER_ASSERT(action_mgr_);
    output_->insert(std::make_shared<ActionManagerOutput>(action_mgr_));
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
