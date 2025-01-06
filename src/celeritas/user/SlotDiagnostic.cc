//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SlotDiagnostic.cc
//---------------------------------------------------------------------------//
#include "SlotDiagnostic.hh"

#include <fstream>
#include <nlohmann/json.hpp>

#include "corecel/Macros.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/ParticleParamsOutput.hh"

#include "detail/SlotDiagnosticExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct SlotDiagnostic::State final : AuxStateInterface
{
    std::ofstream outfile;
    std::vector<int> buffer;

    State() = default;
    // NOLINTNEXTLINE(performance-noexcept-move-constructor)
    CELER_DEFAULT_MOVE_DELETE_COPY(State);
    ~State() final
    {
        CELER_LOG_LOCAL(debug) << "Closing slot diagnostic file";
    }
};

//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<SlotDiagnostic>
SlotDiagnostic::make_and_insert(CoreParams const& core,
                                std::string filename_base)
{
    ActionRegistry& actions = *core.action_reg();
    AuxParamsRegistry& aux = *core.aux_reg();
    auto result = std::make_shared<SlotDiagnostic>(actions.next_id(),
                                                   aux.next_id(),
                                                   std::move(filename_base),
                                                   core.max_streams(),
                                                   core.particle());
    actions.insert(result);
    aux.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with IDs and filename base.
 *
 * This also writes to the "metadata" JSON suffix.
 */
SlotDiagnostic::SlotDiagnostic(ActionId action_id,
                               AuxId aux_id,
                               std::string filename_base,
                               size_type num_streams,
                               std::shared_ptr<ParticleParams const> particle)
    : sad_{action_id, "slot-diagnostic", "track slot properties"}
    , aux_id_{aux_id}
    , filename_base_{std::move(filename_base)}
{
    CELER_EXPECT(aux_id_);
    CELER_EXPECT(!filename_base_.empty());
    CELER_EXPECT(num_streams > 0);

    // Write metadata file with particle names
    std::string filename = filename_base_ + "metadata.json";
    std::ofstream outfile(filename, std::ios::out | std::ios::trunc);
    CELER_VALIDATE(outfile, << "failed to open file at '" << filename << "'");
    outfile <<
        [&particle, &num_streams] {
            JsonPimpl json_wrap;
            ParticleParamsOutput{particle}.output(&json_wrap);
            nlohmann::json j = {
                {"id", "particle"},
                {"metadata", std::move(json_wrap.obj)},
                {"num_streams", num_streams},
            };
            return j;
        }()
            .dump(0);
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
    state.outfile << j.dump() << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
