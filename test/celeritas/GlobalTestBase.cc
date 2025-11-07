//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GlobalTestBase.cc
//---------------------------------------------------------------------------//
#include "GlobalTestBase.hh"

#include <fstream>
#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/random/params/RngParams.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/SurfaceParams.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/phys/GeneratorRegistry.hh"
#include "celeritas/track/ExtendFromPrimariesAction.hh"
#include "celeritas/track/StatusChecker.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
GlobalTestBase::GlobalTestBase()
{
#ifndef __APPLE__
    // ROOT injects handlers simply by being linked on Linux systems
    ScopedRootErrorHandler::disable_signal_handler();
#endif

    // Create output registry
    output_reg_ = std::make_shared<OutputRegistry>();
}

//---------------------------------------------------------------------------//
GlobalTestBase::~GlobalTestBase()
{
    if (this->HasFailure() && output_reg_ && !output_reg_->empty())
    {
        try
        {
            std::string destination = this->make_unique_filename(".out.json");
            std::cerr << "Writing diagnostic output because test failed\n";
            this->write_output();
        }
        catch (std::exception const& e)
        {
            std::cerr << "Failed to write diagnostics: " << e.what();
        }
    }
    // Reset global volumes that we set
    celeritas::global_volumes(nullptr);
}

//---------------------------------------------------------------------------//
/*!
 * Add primaries to be generated.
 */
auto GlobalTestBase::primaries_action() -> SPConstPrimariesAction const&
{
    if (!primaries_action_)
    {
        primaries_action_
            = ExtendFromPrimariesAction::find_action(*this->core());
        CELER_ASSERT(primaries_action_);
    }
    return primaries_action_;
}

//---------------------------------------------------------------------------//
/*!
 * Add primaries to be generated.
 */
void GlobalTestBase::insert_primaries(CoreStateInterface& state,
                                      SpanConstPrimary primaries)
{
    this->primaries_action()->insert(*core_, state, primaries);
}

//---------------------------------------------------------------------------//
/*!
 * Build a new geometry via LazyGeantGeoManager.
 */
auto GlobalTestBase::build_geo_from_geant(SPConstGeantGeo const& geant_geo) const
    -> SPConstGeoI
{
    CELER_EXPECT(geant_geo);
    return CoreGeoParams::from_geant(geant_geo);
}

//---------------------------------------------------------------------------//
/*!
 * Build a new geometry via LazyGeantGeoManager (fallback when no Geant4).
 */
auto GlobalTestBase::build_geo_from_gdml(std::string const& filename) const
    -> SPConstGeoI
{
    CELER_EXPECT(!CELERITAS_USE_GEANT4);
    // ORANGE should be able to handle this, VecGeom can use VGDML
    return CoreGeoParams::from_gdml(filename);
}

//---------------------------------------------------------------------------//
/*!
 * Do not insert StatusChecker.
 */
void GlobalTestBase::disable_status_checker()
{
    CELER_VALIDATE(!core_,
                   << "disable_status_checker cannot be called after core "
                      "params have been created");
    insert_status_checker_ = false;
}

//---------------------------------------------------------------------------//
//! Construct geometry, volumes, surfaces
auto GlobalTestBase::build_geometry() -> SPConstCoreGeo
{
    // Construct core geo
    auto core_geo
        = std::dynamic_pointer_cast<CoreGeoParams const>(this->lazy_geo());
    CELER_ASSERT(core_geo);

    // Get model for constructing volumes/surfaces
    std::shared_ptr<GeoParamsInterface const> model_geo{core_geo};
    if (auto ggeo = this->geant_geo())
    {
        // Load geometry, surfaces, regions from Geant4 world pointer
        model_geo = std::move(ggeo);
    }

    auto mi = model_geo->make_model_input();
    volume_ = std::make_shared<VolumeParams>(mi.volumes);
    celeritas::global_volumes(volume_);
    surface_ = std::make_shared<SurfaceParams>(mi.surfaces, *volume_);

    return core_geo;
}

//---------------------------------------------------------------------------//
auto GlobalTestBase::build_rng() const -> SPConstRng
{
    return std::make_shared<RngParams>(20220511);
}

//---------------------------------------------------------------------------//
auto GlobalTestBase::build_action_reg() const -> SPActionRegistry
{
    return std::make_shared<ActionRegistry>();
}

//---------------------------------------------------------------------------//
auto GlobalTestBase::build_aux_reg() const -> SPUserRegistry
{
    return std::make_shared<AuxParamsRegistry>();
}

//---------------------------------------------------------------------------//
auto GlobalTestBase::build_optical_action_reg() const -> SPActionRegistry
{
    return std::make_shared<ActionRegistry>();
}

//---------------------------------------------------------------------------//
optical::CoreParams::Input GlobalTestBase::optical_params_input()
{
    optical::CoreParams::Input inp;
    inp.geometry = this->geometry();
    inp.material = this->optical_material();
    inp.rng = this->rng();
    inp.surface = this->core()->surface();
    inp.action_reg = this->optical_action_reg();
    inp.gen_reg = std::make_shared<GeneratorRegistry>();
    inp.physics = this->optical_physics();
    inp.surface_physics = this->optical_surface_physics();
    inp.cherenkov = this->cherenkov();
    inp.scintillation = this->scintillation();

    CELER_ENSURE(inp);
    return inp;
}

//---------------------------------------------------------------------------//
auto GlobalTestBase::build_optical_params() -> SPOpticalParams
{
    return std::make_shared<optical::CoreParams>(this->optical_params_input());
}

//---------------------------------------------------------------------------//
auto GlobalTestBase::build_core() -> SPConstCore
{
    CoreParams::Input inp;
    inp.geometry = this->geometry();
    if (!surface_)
    {
        surface_ = std::make_shared<SurfaceParams>();
    }
    if (!volume_)
    {
        volume_ = std::make_shared<VolumeParams>();
    }

    inp.cutoff = this->cutoff();
    inp.geomaterial = this->geomaterial();
    inp.init = this->init();
    inp.material = this->material();
    inp.particle = this->particle();
    inp.physics = this->physics();
    inp.rng = this->rng();
    inp.sim = this->sim();
    inp.surface = surface_;
    inp.volume = volume_;
    inp.wentzel = this->wentzel();

    inp.action_reg = this->action_reg();
    inp.output_reg = this->output_reg();
    inp.aux_reg = this->aux_reg();
    CELER_ASSERT(inp);

    // Build along-step action to add to the stepping loop
    auto&& along_step = this->along_step();
    CELER_ASSERT(along_step);

    if (insert_status_checker_)
    {
        // For unit testing, add status checker
        auto status_checker = std::make_shared<StatusChecker>(
            inp.action_reg->next_id(), inp.aux_reg->next_id());
        inp.action_reg->insert(status_checker);
        inp.aux_reg->insert(status_checker);
    }

    return std::make_shared<CoreParams>(std::move(inp));
}

//---------------------------------------------------------------------------//
void GlobalTestBase::write_output()
{
    std::string filename = this->make_unique_filename(".out.json");
    std::ofstream ofs(filename);
    CELER_VALIDATE(ofs, << "failed to open output file at " << filename);

    // Print with pretty indentation
    {
        JsonPimpl json_wrap;
        this->output_reg()->output(&json_wrap);
        ofs << json_wrap.obj.dump(1) << '\n';
    }

    CELER_LOG(info) << "Wrote output to " << filename;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
