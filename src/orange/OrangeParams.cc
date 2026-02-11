//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParams.cc
//---------------------------------------------------------------------------//
#include "OrangeParams.hh"

#include <fstream>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/data/Collection.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/BoundingBox.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/VolumeParams.hh"

#include "OrangeData.hh"  // IWYU pragma: associated
#include "OrangeInput.hh"
#include "OrangeInputIO.json.hh"  // IWYU pragma: keep
#include "OrangeTypes.hh"
#include "g4org/Converter.hh"
#include "univ/detail/LogicStack.hh"

#include "detail/ConvertLogic.hh"
#include "detail/DepthCalculator.hh"
#include "detail/RectArrayInserter.hh"
#include "detail/UnitInserter.hh"
#include "detail/UniverseInserter.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Utility for counters, labels
template<class T>
struct Components
{
    T universe{};
    T volume{};
    T surface{};
};

using ComponentLabels = Components<std::vector<Label>>;

//---------------------------------------------------------------------------//
//! Determine number of label components and reserve space for them
ComponentLabels make_reserved_label_vecs(OrangeInput const& input)
{
    Components<std::size_t> sizes;
    sizes.universe = input.universes.size();

    for (auto const& u : input.universes)
    {
        std::visit(Overload{
                       [&sizes](UnitInput const& i) {
                           using InserterT = detail::UnitInserter;
                           sizes.surface += InserterT::num_surfaces(i);
                           sizes.volume += InserterT::num_volumes(i);
                       },
                       [&sizes](RectArrayInput const& i) {
                           using InserterT = detail::RectArrayInserter;
                           sizes.surface += InserterT::num_surfaces(i);
                           sizes.volume += InserterT::num_volumes(i);
                       },
                   },
                   u);
    }

    CELER_LOG(debug) << "Allocating labels for " << sizes.universe
                     << " universes with " << sizes.volume
                     << " impl volumes and " << sizes.surface
                     << " impl surfaces";

    Components<std::vector<Label>> result;
    result.universe.reserve(sizes.universe);
    result.volume.reserve(sizes.volume);
    result.surface.reserve(sizes.surface);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Build by loading a GDML file.
 *
 * This mode is incompatible with having an existing run manager.
 */
std::shared_ptr<OrangeParams>
OrangeParams::from_gdml(std::string const& filename)
{
    CELER_VALIDATE(celeritas::global_geant_geo().expired(),
                   << "cannot load Geant4 geometry into ORANGE from a "
                      "file name: a global Geant4 geometry already "
                      "exists");

    if (!CELERITAS_USE_GEANT4)
    {
        CELER_LOG(warning) << "Using ORANGE geometry with GDML suffix "
                              "when Geant4 is disabled: trying "
                              "`.org.json` instead";
        CELER_VALIDATE(ends_with(filename, ".gdml"),
                       << "invalid extension for GDML file '" << filename
                       << "'");

        std::string json_filename(filename.begin(), filename.end() - 5);
        json_filename += ".org.json";
        return OrangeParams::from_json(json_filename);
    }

    // Load temporarily and convert
    auto temp_geant_geo = GeantGeoParams::from_gdml(filename);
    return OrangeParams::from_geant(temp_geant_geo);
}

//---------------------------------------------------------------------------//
/*!
 * Build from a Geant4 world.
 */
std::shared_ptr<OrangeParams>
OrangeParams::from_geant(std::shared_ptr<GeantGeoParams const> const& geo,
                         SPConstVolumes volumes)
{
    CELER_EXPECT(geo);
    CELER_EXPECT(volumes);
    CELER_EXPECT(!volumes->empty());

    ScopedProfiling profile_this{"orange-load-geant"};

    // Set up options for debug output
    inp::OrangeGeoFromGeant opts;
    if (auto opt_filename = celeritas::getenv("G4ORG_OPTIONS");
        !opt_filename.empty())
    {
        std::ifstream infile{opt_filename};
        CELER_VALIDATE(infile,
                       << "failed to open g4org option file at '"
                       << opt_filename << "'");
        try
        {
            infile >> opts;
            CELER_LOG(debug) << "Loaded ORANGE conversion options: " << opts;
        }
        catch (std::exception const& e)
        {
            CELER_LOG(critical) << "Failed to load options from "
                                << opt_filename << ": " << e.what();
        }
    }

    // Convert G4 geometry to ORANGE input data structure
    auto result = g4org::Converter{std::move(opts)}(*geo, *volumes).input;

    return std::make_shared<OrangeParams>(std::move(result),
                                          std::move(volumes));
}

//---------------------------------------------------------------------------//
/*!
 * Build from a Geant4 world (no volumes available?).
 */
std::shared_ptr<OrangeParams>
OrangeParams::from_geant(std::shared_ptr<GeantGeoParams const> const& geo)
{
    CELER_EXPECT(geo);
    SPConstVolumes volumes = celeritas::global_volumes().lock();
    if (!volumes)
    {
        CELER_LOG(debug) << "Constructing global volumes from GeantGeoParams";
        volumes
            = std::make_shared<VolumeParams>(geo->make_model_input().volumes);
        celeritas::global_volumes(volumes);
    }
    return OrangeParams::from_geant(geo, std::move(volumes));
}

//---------------------------------------------------------------------------//
/*!
 * Build from a JSON input.
 */
std::shared_ptr<OrangeParams>
OrangeParams::from_json(std::string const& filename)
{
    CELER_LOG(info) << "Loading ORANGE geometry from JSON at " << filename;
    ScopedTimeLog scoped_time;
    ScopedProfiling profile_this{"orange-load-json"};

    OrangeInput result;

    std::ifstream infile(filename);
    CELER_VALIDATE(infile,
                   << "failed to open geometry at '" << filename << '\'');
    // Use the `from_json` defined in OrangeInputIO.json to read the JSON input
    nlohmann::json::parse(infile).get_to(result);

    return std::make_shared<OrangeParams>(std::move(result));
}

//---------------------------------------------------------------------------//
/*!
 * Advanced usage: construct from explicit host data.
 */
OrangeParams::OrangeParams(OrangeInput&& input)
    : OrangeParams{std::move(input), nullptr}
{
}

//---------------------------------------------------------------------------//
/*!
 * Advanced usage: construct from explicit host data and volumes.
 */
OrangeParams::OrangeParams(OrangeInput&& input, SPConstVolumes&& volumes)
    : volumes_{std::move(volumes)}
{
    CELER_VALIDATE(input, << "input geometry is incomplete");

    ScopedProfiling profile_this{"orange-construct"};
    ScopedMem record_mem("orange.finalize_runtime");
    CELER_LOG(debug) << "Merging runtime data"
                     << (celeritas::device() ? " and copying to GPU" : "");
    ScopedTimeLog scoped_time;

    // First, preprocess the input logic expressions to match the tracker
    detail::convert_logic(input, orange_tracking_logic);
    CELER_ASSERT(input.logic == orange_tracking_logic);

    // Save global bounding box
    bbox_ = [&input] {
        auto& global = input.universes[orange_global_univ.unchecked_get()];
        CELER_VALIDATE(std::holds_alternative<UnitInput>(global),
                       << "global universe is not a SimpleUnit");
        return std::get<UnitInput>(global).bbox;
    }();

    // Create host data for construction, setting tolerances first
    HostVal<OrangeParamsData> host_data;
    host_data.scalars.tol = input.tol;
    host_data.scalars.num_univ_levels
        = detail::DepthCalculator{input.universes}();
    host_data.scalars.num_vol_levels = volumes_ ? volumes_->num_volume_levels()
                                                : 0;

    // Insert all universes
    {
        ComponentLabels labels = make_reserved_label_vecs(input);

        detail::UniverseInserter insert_universe_base{volumes_,
                                                      &labels.universe,
                                                      &labels.surface,
                                                      &labels.volume,
                                                      &host_data};
        Overload insert_universe{
            detail::UnitInserter{
                &insert_universe_base, &host_data, &(input.construction_opts)},
            detail::RectArrayInserter{&insert_universe_base, &host_data}};

        for (auto&& u : input.universes)
        {
            std::visit(insert_universe, std::move(u));
        }

        univ_labels_ = UniverseMap{"universe", std::move(labels.universe)};
        impl_surf_labels_
            = SurfaceMap{"impl surface", std::move(labels.surface)};
        impl_vol_labels_
            = ImplVolumeMap{"impl volume", std::move(labels.volume)};
    }

    // Clear captured input since we've consumed and modified it
    std::move(input) = {};

    // Simple safety if all SimpleUnits have simple safety and no RectArrays
    // are present
    supports_safety_
        = std::all_of(
              host_data.simple_units[AllItems<SimpleUnitRecord>()].begin(),
              host_data.simple_units[AllItems<SimpleUnitRecord>()].end(),
              [](SimpleUnitRecord const& su) { return su.simple_safety; })
          && host_data.rect_arrays.empty();

    // Update scalars *after* loading all units
    CELER_VALIDATE(
        orange_tracking_logic == LogicNotation::infix
            || host_data.scalars.max_csg_levels
                   <= detail::LogicStack::capacity(),
        << R"(input geometry has at least one volume with a CSG tree depth of)"
        << host_data.scalars.max_csg_levels
        << ", but the logic stack is limited to a depth of "
        << detail::LogicStack::capacity());

    // Save pointers for debug output
    host_data.scalars.host_geo_params = this;
    host_data.scalars.host_volume_params = volumes_.get();

    // Define one-to-one mappings if IDs are empty (i.e., JSON-loaded geometry)
    // see make_model_input
    if (host_data.volume_ids.empty())
    {
        resize(&host_data.volume_ids, impl_vol_labels_.size());
        resize(&host_data.volume_instance_ids, impl_vol_labels_.size());
        for (auto ivi_id : range(ImplVolumeId{impl_vol_labels_.size()}))
        {
            host_data.volume_ids[ivi_id] = id_cast<VolumeId>(ivi_id.get());
            host_data.volume_instance_ids[ivi_id]
                = id_cast<VolumeInstanceId>(ivi_id.get());
        }
    }

    CELER_ASSERT(host_data.volume_ids.size() == impl_vol_labels_.size());
    CELER_ASSERT(host_data.volume_instance_ids.size()
                 == impl_vol_labels_.size());

    // Construct device values and device/host references
    CELER_ASSERT(host_data);
    data_ = ParamsDataStore{std::move(host_data)};

    CELER_ENSURE(impl_surf_labels_ && univ_labels_ && impl_vol_labels_);
    CELER_ENSURE(data_);
    CELER_ENSURE(impl_vol_labels_.size() > 0);
    CELER_ENSURE(bbox_);
}

//---------------------------------------------------------------------------//
/*!
 * Create model parameters corresponding to our internal representation.
 *
 * This currently just maps volumes to volume instances to allow tests to pass
 * when Geant4 is disabled and the real volume hierarchy is unavailable.
 */
inp::Model OrangeParams::make_model_input() const
{
    CELER_LOG(info) << R"(Generating fake model input for unit tests)";

    inp::Model result;
    inp::Volumes& v = result.volumes;
    v.volumes.resize(impl_vol_labels_.size());
    v.volume_instances.resize(v.volumes.size());

    // Process each volume: same for impl, vol, vol inst
    for (auto idx : range(impl_vol_labels_.size()))
    {
        auto const& label = impl_vol_labels_.at(ImplVolumeId{idx});
        if (label.name.empty()
            || (label.name.front() == '[' && label.name.back() == ']'))
        {
            // Skip pretend volumes
            continue;
        }

        v.volumes[idx].label = label;
        v.volumes[idx].material = GeoMatId{0};
        v.volume_instances[idx].label = label;
        v.volume_instances[idx].volume = id_cast<VolumeId>(idx);
    }

    v.world = VolumeId{0};
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor to define vtable externally.
 *
 * Needed due to a buggy LLVM optimization that causes dynamic-casts in
 * downstream libraries to fail: see
 * https://github.com/celeritas-project/celeritas/pull/1436 .
 */
OrangeParams::~OrangeParams() = default;

template class ParamsDataStore<OrangeParamsData>;
template class ParamsDataInterface<OrangeParamsData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
