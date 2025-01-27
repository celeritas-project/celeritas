//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.cc
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include <type_traits>
#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/io/OutputRegistry.hh"  // IWYU pragma: keep
#include "corecel/sys/ActionRegistry.hh"  // IWYU pragma: keep
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep

#include "ParticleTallyData.hh"

#include "detail/StepDiagnosticExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<StepDiagnostic>
StepDiagnostic::make_and_insert(CoreParams const& core, size_type max_step_bin)
{
    ActionRegistry& actions = *core.action_reg();
    OutputRegistry& out = *core.output_reg();
    auto result = std::make_shared<StepDiagnostic>(
        actions.next_id(), core.particle(), max_step_bin, core.max_streams());
    actions.insert(result);
    out.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with particle data.
 */
StepDiagnostic::StepDiagnostic(ActionId id,
                               SPConstParticle particle,
                               size_type max_step_bin,
                               size_type num_streams)
    : id_(id), num_streams_(num_streams)
{
    CELER_EXPECT(id_);
    CELER_EXPECT(particle);
    CELER_EXPECT(num_streams_ > 0);

    CELER_VALIDATE(max_step_bin > 0,
                   << "nonpositive step diagnostic 'max' bin");

    // Add two extra bins for underflow and overflow
    HostVal<ParticleTallyParamsData> host_params;
    host_params.num_bins = max_step_bin + 2;
    host_params.num_particles = particle->size();
    store_ = {std::move(host_params), num_streams_};

    CELER_ENSURE(store_);
}

//---------------------------------------------------------------------------//
//! Default destructor
StepDiagnostic::~StepDiagnostic() = default;

//---------------------------------------------------------------------------//
/*!
 * Execute action with host data.
 */
void StepDiagnostic::step(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::StepDiagnosticExecutor{
            store_.params<MemSpace::native>(),
            store_.state<MemSpace::native>(state.stream_id(),
                                           this->state_size())});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void StepDiagnostic::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Get a long description of the action.
 */
std::string_view StepDiagnostic::description() const
{
    return "accumulate total step counters";
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void StepDiagnostic::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    auto obj = json::object();

    obj["steps"] = this->calc_steps();
    obj["_index"] = {"particle", "num_steps"};

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
/*!
 * Get the diagnostic results accumulated over all streams.
 */
auto StepDiagnostic::calc_steps() const -> VecVecCount
{
    // Get the raw data accumulated over all host/device streams
    std::vector<size_type> counts(this->state_size(), 0);
    accumulate_over_streams(
        store_, [](auto& state) { return state.counts; }, &counts);

    auto const& params = store_.params<MemSpace::host>();

    VecVecCount result(params.num_particles);
    for (auto i : range(result.size()))
    {
        auto start = counts.begin() + i * params.num_bins;
        CELER_ASSERT(start + params.num_bins <= counts.end());
        result[i] = {start, start + params.num_bins};
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Size of diagnostic state data (number of bins times number of particles)
 */
size_type StepDiagnostic::state_size() const
{
    auto const& params = store_.params<MemSpace::host>();
    return params.num_bins * params.num_particles;
}

//---------------------------------------------------------------------------//
/*!
 * Reset diagnostic results.
 */
void StepDiagnostic::clear()
{
    apply_to_all_streams(
        store_, [](auto& state) { fill(size_type(0), &state.counts); });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
