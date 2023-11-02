//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/FieldDiagnostic.cc
//---------------------------------------------------------------------------//
#include "FieldDiagnostic.hh"

#include <type_traits>
#include <utility>
#include <vector>

#include "celeritas_config.h"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/JsonPimpl.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/user/FieldDiagnosticData.hh"

#include "detail/FieldDiagnosticExecutor.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "corecel/io/LabelIO.json.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the number of bins.
 */
FieldDiagnostic::FieldDiagnostic(ActionId id,
                                 UniformGridData energy,
                                 size_type num_substep_bins,
                                 size_type num_streams)
    : id_(id), num_streams_(num_streams)
{
    CELER_EXPECT(id_);
    CELER_EXPECT(energy);
    CELER_EXPECT(num_substep_bins > 0);
    CELER_EXPECT(num_streams_ > 0);

    HostVal<FieldDiagnosticParamsData> host_params;

    // Add two extra bins (for underflow and overflow)
    host_params.energy = energy;
    host_params.num_substep_bins = num_substep_bins + 2;

    store_ = {std::move(host_params), num_streams_};

    CELER_ENSURE(store_);
}

//---------------------------------------------------------------------------//
//! Default destructor
FieldDiagnostic::~FieldDiagnostic() = default;

//---------------------------------------------------------------------------//
/*!
 * Execute action with host data.
 */
void FieldDiagnostic::execute(CoreParams const& params,
                              CoreStateHost& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::FieldDiagnosticExecutor{
            store_.params<MemSpace::native>(),
            store_.state<MemSpace::native>(state.stream_id(),
                                           this->state_size())});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void FieldDiagnostic::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Get a long description of the action.
 */
std::string FieldDiagnostic::description() const
{
    return "record number of field substeps vs. track energy";
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void FieldDiagnostic::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto obj = json::object();

    UniformGridData const& grid = store_.params<MemSpace::host>().energy;
    std::vector<real_type> energy(grid.size);
    energy.front() = std::exp(grid.front);
    energy.back() = std::exp(grid.back);
    real_type curr = grid.front;
    for (size_type i = 1; i < energy.size() - 1; ++i)
    {
        curr += grid.delta;
        energy[i] = std::exp(curr);
    }

    obj["energy"] = std::move(energy);
    obj["substeps"] = this->calc_num_iter();
    obj["_index"] = {"energy", "num_substeps"};

    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the diagnostic results accumulated over all streams.
 */
auto FieldDiagnostic::calc_num_iter() const -> VecCount
{
    // Get the raw data accumulated over all host/device streams
    std::vector<size_type> result(this->state_size(), 0);
    accumulate_over_streams(
        store_, [](auto& state) { return state.counts; }, &result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Size of diagnostic state data (number of bins)
 */
size_type FieldDiagnostic::state_size() const
{
    auto const& params = store_.params<MemSpace::host>();
    return (params.energy.size - 1) * params.num_substep_bins;
}

//---------------------------------------------------------------------------//
/*!
 * Reset diagnostic results.
 */
void FieldDiagnostic::clear()
{
    apply_to_all_streams(
        store_, [](auto& state) { fill(size_type(0), &state.counts); });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
