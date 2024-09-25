//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SlotDiagnostic.cc
//---------------------------------------------------------------------------//
#include "SlotDiagnostic.hh"

#include <fstream>
#include <nlohmann/json.hpp>

#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/SlotDiagnosticExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct SlotDiagnostic::State final : AuxStateInterface
{
    std::ofstream outfile;
    std::vector<int> buffer;

    ~State() { CELER_LOG_LOCAL(debug) << "Closing slot diagnostic file"; }
};

//---------------------------------------------------------------------------//
/*!
 * Construct with IDs and filename base.
 */
SlotDiagnostic::SlotDiagnostic(ActionId action_id,
                               AuxId aux_id,
                               std::string filename_base)
    : sad_{action_id, "slot-diagnostic", "track slot properties"}
    , aux_id_{aux_id}
    , filename_base_{std::move(filename_base)}
{
    CELER_EXPECT(aux_id_);
    CELER_EXPECT(!filename_base_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto SlotDiagnostic::create_state(MemSpace,
                                  StreamId id,
                                  size_type size) const -> UPState
{
    auto result = std::make_unique<State>();
    result->buffer.resize(size);

    std::string filename = filename_base_ + std::to_string(id.get()) + ".jsonl";
    result->outfile.open(filename, std::ios::out | std::ios::trunc);
    CELER_VALIDATE(result->outfile,
                   << "failed to open file at '" << filename << "'");
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Execute with with the last action's ID and the state.
 *
 * This must be called after both \c create_state and \c begin_run .
 */
void SlotDiagnostic::step(CoreParams const& params, CoreStateHost& state) const
{
    auto buffer = this->get_host_buffer(state.aux());
    CELER_ASSERT(buffer.size() == state.size());

    // Copy IDs directly into buffer
    launch_core(this->label(),
                params,
                state,
                TrackExecutor{params.ptr<MemSpace::native>(),
                              state.ptr(),
                              detail::SlotDiagnosticExecutor{
                                  ObserverPtr{buffer.data()}}});

    // Write IDs to
    this->write_buffer(state.aux());
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void SlotDiagnostic::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Obtain CPU storage space for writing the values in each thread.
 */
Span<int> SlotDiagnostic::get_host_buffer(AuxStateVec& aux_state) const
{
    auto& state = get<State>(aux_state, aux_id_);
    return make_span(state.buffer);
}

//---------------------------------------------------------------------------//
/*!
 * Write the buffer as a JSON line.
 */
void SlotDiagnostic::write_buffer(AuxStateVec& aux_state) const
{
    auto& state = get<State>(aux_state, aux_id_);
    CELER_EXPECT(state.outfile);

    // Write the buffer as a line of JSON output
    nlohmann::json j = state.buffer;
    state.outfile << j.dump(0) << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
