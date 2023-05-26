//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionDiagnostic.cc
//---------------------------------------------------------------------------//
#include "ActionDiagnostic.hh"

#include <mutex>

#include "corecel/Assert.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "detail/ActionDiagnosticImpl.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "corecel/io/LabelIO.json.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the action ID.
 *
 * Other required attributes are deferred until beginning-of-run.
 */
ActionDiagnostic::ActionDiagnostic(ActionId id) : id_(id)
{
    CELER_EXPECT(id_);
}

//---------------------------------------------------------------------------//
//! Default destructor
ActionDiagnostic::~ActionDiagnostic() = default;

//---------------------------------------------------------------------------//
/*!
 * Build the storage for diagnostic parameters and stream-dependent states.
 *
 * This must be done lazily (after construction!) because the action diagnostic
 * will be created *before* all actions are defined in the \c ActionRegistry.
 */
void ActionDiagnostic::begin_run(CoreParams const& params, CoreStateHost&)
{
    return this->begin_run_impl(params);
}

//---------------------------------------------------------------------------//
/*!
 * Build the storage for diagnostic parameters and stream-dependent states.
 *
 * Since the stream store is (currently) lazily constructed, we can call the
 * same begin_run as host.
 */
void ActionDiagnostic::begin_run(CoreParams const& params, CoreStateDevice&)
{
    return this->begin_run_impl(params);
}

//---------------------------------------------------------------------------//
/*!
 * Execute action with host data.
 */
void ActionDiagnostic::execute(CoreParams const& params,
                               CoreStateHost& state) const
{
    MultiExceptionHandler capture_exception;
    auto launch = make_active_track_launcher(
        *params.ptr<MemSpace::native>(),
        *state.ptr(),
        detail::tally_action,
        store_.params<MemSpace::host>(),
        store_.state<MemSpace::host>(state.stream_id(), this->state_size()));
#pragma omp parallel for
    for (ThreadId::size_type i = 0; i < state.size(); ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            launch(ThreadId{i}),
            capture_exception,
            KernelContextException(params.ref<MemSpace::host>(),
                                   state.ref(),
                                   ThreadId{i},
                                   this->label()));
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
/*!
 * Get a long description of the action.
 */
std::string ActionDiagnostic::description() const
{
    return "accumulate post-step action counters";
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void ActionDiagnostic::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto obj = json::object();

    obj["actions"] = this->calc_actions();
    obj["_index"] = {"particle", "action"};

    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the nonzero diagnostic results accumulated over all streams.
 *
 * This builds a map of particle/action combinations to the number of
 * occurances over all events.
 */
auto ActionDiagnostic::calc_actions_map() const -> MapStringCount
{
    // Counts indexed as [particle][action]
    auto particle_vec = this->calc_actions();

    // Get shared pointers from the weak pointers
    auto sp_action_reg = action_reg_.lock();
    auto sp_particle = particle_.lock();
    CELER_ASSERT(sp_action_reg && sp_particle);

    // Map particle ID/action ID to name and store counts
    MapStringCount result;
    for (auto particle : range(ParticleId(particle_vec.size())))
    {
        auto const& action_vec = particle_vec[particle.get()];
        for (auto action : range(ActionId(action_vec.size())))
        {
            if (action_vec[action.get()] > 0)
            {
                std::string label = sp_action_reg->id_to_label(action) + " "
                                    + sp_particle->id_to_label(particle);
                result.insert({std::move(label), action_vec[action.get()]});
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the diagnostic results accumulated over all streams.
 */
auto ActionDiagnostic::calc_actions() const -> VecVecCount
{
    if (!store_)
    {
        CELER_LOG(error) << "Tried to access action counters before executing "
                            "any actions";
        return {};
    }

    // Get the raw data accumulated over all host/device streams
    VecCount counts(this->state_size(), 0);
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
 * Diagnostic state data size (number of particles times number of actions).
 */
size_type ActionDiagnostic::state_size() const
{
    CELER_EXPECT(store_);

    auto const& params = store_.params<MemSpace::host>();
    return params.num_bins * params.num_particles;
}

//---------------------------------------------------------------------------//
/*!
 * Reset diagnostic results.
 */
void ActionDiagnostic::clear()
{
    CELER_EXPECT(store_);

    apply_to_all_streams(
        store_, [](auto& state) { fill(size_type(0), &state.counts); });
}

//---------------------------------------------------------------------------//
/*!
 * Build the storage for diagnostic parameters and stream-dependent states.
 *
 * This must be done lazily (after construction!) because the action diagnostic
 * will be created *before* all actions are defined in the \c ActionRegistry.
 */
void ActionDiagnostic::begin_run_impl(CoreParams const& params)
{
    if (!store_)
    {
        static std::mutex initialize_mutex;
        std::lock_guard<std::mutex> scoped_lock{initialize_mutex};

        if (!store_)
        {
            action_reg_ = params.action_reg();
            particle_ = params.particle();

            HostVal<ParticleTallyParamsData> host_params;
            host_params.num_bins = params.action_reg()->num_actions();
            host_params.num_particles = params.particle()->size();
            store_ = {std::move(host_params), params.max_streams()};
        }
    }
    CELER_ENSURE(store_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
