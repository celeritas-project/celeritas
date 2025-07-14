//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParams.cc
//---------------------------------------------------------------------------//
#include "OrangeParams.hh"

#include <fstream>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/BoundingBox.hh"
#include "geocel/GeantGeoParams.hh"

#include "OrangeData.hh"  // IWYU pragma: associated
#include "OrangeInput.hh"
#include "OrangeInputIO.json.hh"
#include "OrangeTypes.hh"
#include "g4org/Converter.hh"
#include "univ/detail/LogicStack.hh"

#include "detail/DepthCalculator.hh"
#include "detail/RectArrayInserter.hh"
#include "detail/UnitInserter.hh"
#include "detail/UniverseInserter.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Load a geometry from the given JSON file.
 */
OrangeInput input_from_json(std::string filename)
{
    CELER_LOG(info) << "Loading ORANGE geometry from JSON at " << filename;
    ScopedTimeLog scoped_time;

    OrangeInput result;

    std::ifstream infile(filename);
    CELER_VALIDATE(infile,
                   << "failed to open geometry at '" << filename << '\'');
    // Use the `from_json` defined in OrangeInputIO.json to read the JSON input
    nlohmann::json::parse(infile).get_to(result);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Load a geometry from the given filename.
 */
OrangeInput input_from_file(std::string filename)
{
    if (ends_with(filename, ".gdml"))
    {
        if (CELERITAS_USE_GEANT4
            && CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            // Load with Geant4: must *not* be using run manager
            CELER_VALIDATE(!geant_geo(),
                           << "cannot load Geant4 geometry into ORANGE from a "
                              "file name: a global Geant4 geometry already "
                              "exists");
            GeantGeoParams temp_geant_geo(filename);
            auto result = g4org::Converter{}(temp_geant_geo).input;
            return result;
        }
        else
        {
            CELER_LOG(warning) << "Using ORANGE geometry with GDML suffix "
                                  "when Geant4 conversion is disabled: trying "
                                  "`.org.json` instead";
            filename.erase(filename.end() - 5, filename.end());
            filename += ".org.json";
        }
    }
    else
    {
        CELER_VALIDATE(ends_with(filename, ".json"),
                       << "expected JSON extension for ORANGE input '"
                       << filename << "'");
    }
    return input_from_json(std::move(filename));
}

GeantGeoParams const& get_geant_params(G4VPhysicalVolume const* world)
{
    auto* global_geo = celeritas::geant_geo();
    CELER_VALIDATE(global_geo && global_geo->world() == world,
                   << "world is not the globally defined world");

    return *global_geo;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a JSON file (if JSON is enabled).
 *
 * The JSON format is defined by the SCALE ORANGE exporter (not currently
 * distributed).
 */
OrangeParams::OrangeParams(std::string const& filename)
    : OrangeParams(input_from_file(filename))
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct in-memory from a Geant4 geometry.
 *
 * TODO: expose options? Fix volume mappings?
 */
OrangeParams::OrangeParams(G4VPhysicalVolume const* world)
    : OrangeParams(std::move(g4org::Converter{}(get_geant_params(world)).input))
{
}

//---------------------------------------------------------------------------//
/*!
 * Advanced usage: construct from explicit host data.
 *
 * Volume and surface labels must be unique for the time being.
 */
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
OrangeParams::OrangeParams(OrangeInput&& input)
{
    CELER_VALIDATE(input, << "input geometry is incomplete");

    ScopedProfiling profile_this{"finalize-orange-runtime"};
    ScopedMem record_mem("orange.finalize_runtime");
    CELER_LOG(debug) << "Merging runtime data"
                     << (celeritas::device() ? " and copying to GPU" : "");
    ScopedTimeLog scoped_time;

    // Save global bounding box
    bbox_ = [&input] {
        auto& global = input.universes[orange_global_universe.unchecked_get()];
        CELER_VALIDATE(std::holds_alternative<UnitInput>(global),
                       << "global universe is not a SimpleUnit");
        return std::get<UnitInput>(global).bbox;
    }();

    // Create host data for construction, setting tolerances first
    HostVal<OrangeParamsData> host_data;
    host_data.scalars.tol = input.tol;
    host_data.scalars.max_depth = detail::DepthCalculator{input.universes}();

    // Insert all universes
    {
        std::vector<Label> universe_labels;
        std::vector<Label> surface_labels;
        std::vector<Label> volume_labels;

        detail::UniverseInserter insert_universe_base{
            &universe_labels, &surface_labels, &volume_labels, &host_data};
        Overload insert_universe{
            detail::UnitInserter{&insert_universe_base, &host_data},
            detail::RectArrayInserter{&insert_universe_base, &host_data}};

        for (auto&& u : input.universes)
        {
            std::visit(insert_universe, std::move(u));
        }

        surf_labels_ = SurfaceMap{"surface", std::move(surface_labels)};
        univ_labels_ = UniverseMap{"universe", std::move(universe_labels)};
        vol_labels_ = VolumeMap{"volume", std::move(volume_labels)};
    }

    // Simple safety if all SimpleUnits have simple safety and no RectArrays
    // are present
    supports_safety_
        = std::all_of(
              host_data.simple_units[AllItems<SimpleUnitRecord>()].begin(),
              host_data.simple_units[AllItems<SimpleUnitRecord>()].end(),
              [](SimpleUnitRecord const& su) { return su.simple_safety; })
          && host_data.rect_arrays.empty();

    // Update scalars *after* loading all units
    CELER_VALIDATE(host_data.scalars.max_logic_depth
                       < detail::LogicStack::max_stack_depth(),
                   << "input geometry has at least one volume with a "
                      "logic depth of"
                   << host_data.scalars.max_logic_depth
                   << " (a volume's CSG tree is too deep); but the logic "
                      "stack is limited to a depth of "
                   << detail::LogicStack::max_stack_depth());

    // Construct device values and device/host references
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<OrangeParamsData>{std::move(host_data)};

    CELER_ENSURE(surf_labels_ && univ_labels_ && vol_labels_);
    CELER_ENSURE(data_);
    CELER_ENSURE(vol_labels_.size() > 0);
    CELER_ENSURE(bbox_);
}
//---------------------------------------------------------------------------//
/*!
 * Create model parameters corresponding to our internal representation.
 */
inp::Model OrangeParams::make_model_input() const
{
    CELER_LOG(warning) << "ORANGE cannot yet construct model input";
    inp::Model result;

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

template class CollectionMirror<OrangeParamsData>;
template class ParamsDataInterface<OrangeParamsData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
