//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/ResponseTallyAction.cc
//---------------------------------------------------------------------------//
#include "ResponseTallyAction.hh"

#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/OutputRegistry.hh"  // IWYU pragma: keep
#include "corecel/sys/ActionRegistry.hh"  // IWYU pragma: keep
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/ResponseTallyExecutor.hh"

using namespace celeritas::literals;

namespace celeritas
{
namespace optical
{
namespace
{
//---------------------------------------------------------------------------//
//! Convert a uniform grid to JSON
nlohmann::json grid_to_json(UniformGridData const& grid)
{
    return {{"front", grid.front},
            {"back", grid.back},
            {"num_bins", grid.size - 1}};
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<ResponseTallyAction> ResponseTallyAction::make_and_insert(
    CoreParams const& core, HostParams&& params)
{
    ActionRegistry& actions = *core.action_reg();
    OutputRegistry& out = *core.output_reg();
    auto result = std::make_shared<ResponseTallyAction>(
        actions.next_id(), std::move(params), core.sizes().streams);
    actions.insert(result);
    out.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with ID, tally definition, and number of streams.
 */
ResponseTallyAction::ResponseTallyAction(
    ActionId id, HostParams&& params, size_type num_streams)
    : id_(id)
{
    CELER_EXPECT(id_);
    CELER_VALIDATE(params,
                   << "invalid response tally: grids must be valid and "
                      "consistent with the dimensions");
    CELER_EXPECT(num_streams > 0);

    store_ = {std::move(params), num_streams};
    CELER_ENSURE(store_);
}

//---------------------------------------------------------------------------//
/*!
 * Get a long description of the action.
 */
std::string_view ResponseTallyAction::description() const
{
    return "tally optical photon detector hits by position and time";
}

//---------------------------------------------------------------------------//
/*!
 * Execute action with host data.
 */
void ResponseTallyAction::step(CoreParams const& params,
                               CoreStateHost& state) const
{
    TrackSlotExecutor execute{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::ResponseTallyExecutor{
            store_.params<MemSpace::native>(),
            store_.state<MemSpace::native>(state.stream_id(), state.size())}};
    return launch_action(state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void ResponseTallyAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Write the grid definition and raw hit counts to JSON.
 */
void ResponseTallyAction::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    auto const& params = this->params();

    auto obj = json::object();
    obj["x_grid"] = grid_to_json(params.x_grid);
    obj["y_grid"] = grid_to_json(params.y_grid);
    obj["z_grid"] = grid_to_json(params.z_grid);
    obj["time_grid"] = grid_to_json(params.time_grid);
    obj["dims"]
        = std::vector<size_type>(params.dims.begin(), params.dims.end());
    obj["counts"] = this->calc_counts();
    obj["_index"] = {"detector", "x", "y", "z", "time"};

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
/*!
 * Hit counts accumulated over all streams.
 */
auto ResponseTallyAction::calc_counts() const -> std::vector<size_type>
{
    std::vector<size_type> counts(this->params().num_bins(), 0);
    accumulate_over_streams(
        store_, [](auto& state) { return state.counts; }, &counts);
    return counts;
}

//---------------------------------------------------------------------------//
/*!
 * Hit counts converted to real numbers and normalized by photons created.
 *
 * The number of photons must be provided externally (e.g., from the
 * generator counters) and be nonzero.
 */
auto ResponseTallyAction::calc_response(size_type num_photons) const
    -> std::vector<double>
{
    CELER_VALIDATE(num_photons > 0,
                   << "cannot normalize response tally with no photons");

    auto counts = this->calc_counts();
    std::vector<double> result(counts.size());
    double const norm = 1.0 / static_cast<double>(num_photons);
    for (std::size_t i = 0; i < counts.size(); ++i)
    {
        result[i] = static_cast<double>(counts[i]) * norm;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Reset accumulated counts.
 */
void ResponseTallyAction::clear()
{
    apply_to_all_streams(store_,
                         [](auto& state) { fill(0_sz, &state.counts); });
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
