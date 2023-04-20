//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SimpleCalo.cc
//---------------------------------------------------------------------------//
#include "SimpleCalo.hh"

#include <vector>

#include "celeritas_config.h"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "corecel/cont/LabelIO.json.hh"
#    include "corecel/io/JsonPimpl.hh"
#endif
#include "corecel/data/CollectionAlgorithms.hh"

#include "detail/SimpleCaloImpl.hh"

namespace celeritas
{
namespace {
//---------------------------------------------------------------------------//
//! Helper function for accumulating energy from host and device for all streams
struct SumEnergy
{
    using Energy = SimpleCalo::Energy;

    std::vector<Energy>* result;
    std::vector<Energy> temp_host;

    explicit SumEnergy(std::vector<Energy>* r)
        : result(r)
    {
        CELER_EXPECT(result);
    }

    // Transfer host data
    void operator()(
        SimpleCaloStateData<Ownership::reference, MemSpace::host> const& state)
    {
        CELER_EXPECT(state.energy_deposition.size() == result->size());
        for (auto detid : range(state.energy_deposition.size()))
        {
            *(*result)[detid] += *state.energy_deposition[DetectorId{detid}];
        }
    }

    // Transfer device data
    void operator()(
        SimpleCaloStateData<Ownership::reference, MemSpace::device> const& state)
    {
        CELER_EXPECT(state.energy_deposition.size() == result->size());

        if (temp_host.empty())
        {
            temp_host.resize(result->size());
        }
        copy_to_host(state.energy_deposition, make_span(temp_host));

        for (auto detid : range(state.energy_deposition.size()))
        {
            *(*result)[detid] += *temp_host[detid];
        }
    }
};

//---------------------------------------------------------------------------//
}

//---------------------------------------------------------------------------//
/*!
 * Construct with sensitive regions.
 */
SimpleCalo::SimpleCalo(VecLabel labels, SPConstGeo geo, size_type num_streams)
    : volume_labels_{std::move(labels)}, store_{{}, num_streams}
{
    CELER_EXPECT(!labels.empty());
    CELER_EXPECT(geo);
    CELER_EXPECT(num_streams > 0);

    // XXX map labels to IDs

    CELER_ENSURE(volume_ids_.size() == volume_labels_.size());
    CELER_ENSURE(store_);
}

//---------------------------------------------------------------------------//
/*!
 * Map volume names to detector IDs and exclude tracks with no deposition.
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
 * Process detector tallies (CPU).
 */
void SimpleCalo::process_steps(HostWTFStepState state)
{
    detail::simple_calo_accum(
        state.steps,
        store_.state<MemSpace::host>(state.stream_id, state.steps.size()));
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (GPU).
 */
void SimpleCalo::process_steps(DeviceWTFStepState state)
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
            ids.push_back(vid.get());
        }
        obj["volume_ids"] = std::move(ids);
        obj["volume_labels"] = volume_labels_;
    }

    // Save results
    {
        std::vector<double> edep;
        for (Energy e : this->calc_total_energy_deposition())
        {
            edep.push_back(e.value());
        }
        obj["energy_deposition"] = std::move(edep);
        obj["_units"] = {
            {"energy_deposition", Energy::unit_type::label()},
        };
    }

    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get tallied stream-local data.
 */
template<MemSpace M>
auto SimpleCalo::energy_deposition(StreamId stream_id) const
    -> EnergyRef<M> const&
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
 */
auto SimpleCalo::calc_total_energy_deposition() const -> VecEnergy
{
    VecEnergy result(this->num_detectors(), zero_quantity());

    this->apply_to_all_streams(store_, SumEnergy{&result});
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Reset energy deposition to zero, usually at the start of an event.
 */
void SimpleCalo::reset()
{
    this->apply_to_all_streams(store_,
        [](auto& state) { fill(Energy{0}, &state.energy_deposition); });
}

//---------------------------------------------------------------------------//
template<class S, class F>
void SimpleCalo::apply_to_all_streams(S& store, F&& func)
{
    // Accumulate on host
    for (StreamId s : range(StreamId{store.num_streams()}))
    {
        if (auto* state = store.template state<MemSpace::host>(s))
        {
            func(*state);
        }
    }

    // Accumulate on device into temporary
    for (StreamId s : range(StreamId{store.num_streams()}))
    {
        if (auto* state = store.template state<MemSpace::device>(s))
        {
            func(*state);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
