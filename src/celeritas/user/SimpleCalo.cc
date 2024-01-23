//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SimpleCalo.cc
//---------------------------------------------------------------------------//
#include "SimpleCalo.hh"

#include <functional>
#include <vector>

#include "celeritas_cmake_strings.h"
#include "celeritas_config.h"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Label.hh"
#include "corecel/io/Logger.hh"
#include "orange/GeoParamsInterface.hh"  // IWYU pragma: keep

#include "detail/SimpleCaloImpl.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "corecel/io/JsonPimpl.hh"
#    include "corecel/io/LabelIO.json.hh"
#endif

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
VolumeId find_volume_fuzzy(GeoParamsInterface const& geo, Label const& label)
{
    if (auto id = geo.find_volume(label))
    {
        // Exact match
        return id;
    }

    // Fall back to skipping the extension: look for all possible matches
    auto all_ids = geo.find_volumes(label.name);
    if (all_ids.size() == 1)
    {
        if (!label.ext.empty())
        {
            CELER_LOG(warning)
                << "Failed to exactly match " << celeritas_core_geo
                << " volume from volume '" << label << "'; found '"
                << geo.id_to_label(all_ids.front())
                << "' by omitting the extension";
        }
        return all_ids.front();
    }
    if (all_ids.size() > 1)
    {
        CELER_LOG(warning)
            << "Multiple volumes '"
            << join(all_ids.begin(),
                    all_ids.end(),
                    "', '",
                    [&geo](VolumeId v) { return geo.id_to_label(v); })
            << "' match the name '" << label.name
            << "': returning the last one";
        return all_ids.back();
    }
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with sensitive regions.
 */
SimpleCalo::SimpleCalo(std::string output_label,
                       VecLabel labels,
                       GeoParamsInterface const& geo,
                       size_type num_streams)
    : output_label_{std::move(output_label)}, volume_labels_{std::move(labels)}
{
    CELER_EXPECT(!output_label_.empty());
    CELER_EXPECT(!volume_labels_.empty());
    CELER_EXPECT(num_streams > 0);

    // Map labels to volume IDs
    volume_ids_.resize(volume_labels_.size());
    std::vector<std::reference_wrapper<Label>> missing;
    for (auto i : range(volume_labels_.size()))
    {
        volume_ids_[i] = find_volume_fuzzy(geo, volume_labels_[i]);
        if (!volume_ids_[i])
        {
            missing.emplace_back(volume_labels_[i]);
        }
    }
    CELER_VALIDATE(missing.empty(),
                   << "failed to find " << celeritas_core_geo
                   << " volume(s) for labels '"
                   << join(missing.begin(), missing.end(), "', '"));

    HostVal<SimpleCaloParamsData> host_params;
    host_params.num_detectors = this->num_detectors();
    store_ = {std::move(host_params), num_streams};

    CELER_ENSURE(volume_ids_.size() == volume_labels_.size());
    CELER_ENSURE(store_);
}

//---------------------------------------------------------------------------//
/*!
 * Map volume names to detector IDs and exclude tracks with no deposition.
 */
auto SimpleCalo::filters() const -> Filters
{
    Filters result;

    for (auto didx : range<DetectorId::size_type>(volume_ids_.size()))
    {
        result.detectors[volume_ids_[didx]] = DetectorId{didx};
    }
    result.nonzero_energy_deposition = true;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Only save energy deposition and pre-step volume.
 */
auto SimpleCalo::selection() const -> StepSelection
{
    StepSelection result;
    result.energy_deposition = true;
    result.points[StepPoint::pre].volume_id = true;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (CPU).
 */
void SimpleCalo::process_steps(HostStepState state)
{
    detail::simple_calo_accum(
        state.steps,
        store_.state<MemSpace::host>(state.stream_id, state.steps.size()));
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (GPU).
 */
void SimpleCalo::process_steps(DeviceStepState state)
{
    detail::simple_calo_accum(
        state.steps,
        store_.state<MemSpace::device>(state.stream_id, state.steps.size()));
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void SimpleCalo::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto obj = json::object();

    // Save detector volumes
    {
        std::vector<int> ids;
        for (VolumeId vid : volume_ids_)
        {
            ids.push_back(static_cast<int>(vid.get()));
        }
        obj["volume_ids"] = std::move(ids);
        obj["volume_labels"] = volume_labels_;
    }

    // Save results
    {
        obj["energy_deposition"] = this->calc_total_energy_deposition();
        obj["_units"] = {
            {"energy_deposition", EnergyUnits::label()},
        };
    }

    j->obj = std::move(obj);
#else
    CELER_DISCARD(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get tallied stream-local data.
 */
template<MemSpace M>
auto SimpleCalo::energy_deposition(StreamId stream_id) const
    -> DetectorRef<M> const&
{
    CELER_EXPECT(stream_id < store_.num_streams());
    auto* result = store_.state<M>(stream_id);
    CELER_VALIDATE(result,
                   << "no simple calo state is stored on " << to_cstring(M)
                   << " for stream ID " << stream_id.get());
    return *result;
}

//---------------------------------------------------------------------------//
/*!
 * Get accumulated energy deposition over all streams.
 *
 * This accessor is useful only when integrating over events and over cores. It
 * also integrates over devices in case you happen to be running Celeritas on
 * both host and device for some weird reason.
 *
 * The index in the vector corresponds to the detector ID and is in the same
 * order as the input labels.
 */
auto SimpleCalo::calc_total_energy_deposition() const -> VecReal
{
    VecReal result(this->num_detectors(), real_type{0});

    accumulate_over_streams(
        store_, [](auto& state) { return state.energy_deposition; }, &result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Reset energy deposition to zero, usually at the start of an event.
 */
void SimpleCalo::clear()
{
    apply_to_all_streams(store_, [](auto& state) {
        fill(real_type(0), &state.energy_deposition);
    });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
