//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionDiagnostic.cc
//---------------------------------------------------------------------------//
#include "ActionDiagnostic.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
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
 * Construct with action registry and particle data.
 *
 * The total number of registered actions is needed in the diagnostic params
 * data, so the construction of the stream storage is deferred until all
 * actions have been registered.
 */
ActionDiagnostic::ActionDiagnostic(ActionId id,
                                   WPConstActionRegistry action_reg,
                                   SPConstParticle particle,
                                   size_type num_streams)
    : id_(id)
    , action_reg_(action_reg)
    , particle_(particle)
    , num_streams_(num_streams)
{
    CELER_EXPECT(id_);
    CELER_EXPECT(!action_reg_.expired());
    CELER_EXPECT(particle_);
    CELER_EXPECT(num_streams > 0);
}

//---------------------------------------------------------------------------//
//! Default destructor
ActionDiagnostic::~ActionDiagnostic() = default;

//---------------------------------------------------------------------------//
/*!
 * Execute action with host data.
 */
void ActionDiagnostic::execute(CoreParams const& params,
                               CoreStateHost& state) const
{
    if (!store_)
    {
        this->build_stream_store();
    }
    MultiExceptionHandler capture_exception;
    auto launch = make_active_track_launcher(
        params.ref<MemSpace::native>(),
        state.ref(),
        detail::tally_action,
        store_.params<MemSpace::host>(),
        store_.state<MemSpace::host>(state.stream_id(), this->state_size()));
#pragma omp parallel for
    for (ThreadId::size_type i = 0; i < state.size(); ++i)
    {
        CELER_TRY_HANDLE(launch(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
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

    // Get a shared pointer to the action registry
    auto sp_action_reg = action_reg_.lock();
    CELER_ASSERT(sp_action_reg);

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
                                    + particle_->id_to_label(particle);
                result[label] = action_vec[action.get()];
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
    CELER_EXPECT(store_);

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
void ActionDiagnostic::build_stream_store() const
{
    CELER_EXPECT(!store_);

    auto sp_action_reg = action_reg_.lock();
    CELER_ASSERT(sp_action_reg);

    HostVal<ParticleTallyParamsData> host_params;
    host_params.num_bins = sp_action_reg->num_actions();
    host_params.num_particles = particle_->size();
    store_ = {std::move(host_params), num_streams_};

    CELER_ENSURE(store_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
