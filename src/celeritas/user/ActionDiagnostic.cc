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
#include "celeritas/global/TrackLauncher.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "detail/ActionDiagnosticImpl.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "corecel/io/LabelIO.json.hh"
#endif

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Helper function for accumulating data from host and device for all streams
struct SumCounts
{
    std::vector<size_type>* result;
    std::vector<size_type> temp_host;

    explicit SumCounts(std::vector<size_type>* r) : result(r)
    {
        CELER_EXPECT(result);
    }

    // Transfer host data
    void operator()(ActionDiagnosticStateData<Ownership::reference,
                                              MemSpace::host> const& state)
    {
        using BinId = ItemId<size_type>;

        CELER_EXPECT(state.counts.size() == result->size());
        for (auto bin : range(state.counts.size()))
        {
            (*result)[bin] += state.counts[BinId{bin}];
        }
    }

    // Transfer device data
    void operator()(ActionDiagnosticStateData<Ownership::reference,
                                              MemSpace::device> const& state)
    {
        CELER_EXPECT(state.counts.size() == result->size());

        if (temp_host.empty())
        {
            temp_host.resize(result->size());
        }
        copy_to_host(state.counts, make_span(temp_host));

        for (auto bin : range(state.counts.size()))
        {
            (*result)[bin] += temp_host[bin];
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with action registry and particle data.
 *
 * The total number of registered actions is needed in the diagnostic params
 * data, so the construction of the stream storage is deferred until all
 * actions have been registered.
 */
ActionDiagnostic::ActionDiagnostic(ActionId id,
                                   SPConstActionRegistry action_reg,
                                   SPConstParticle particle,
                                   size_type num_streams)
    : id_(id)
    , action_reg_(action_reg)
    , particle_(particle)
    , num_streams_(num_streams)
    , store_(std::make_unique<StoreT>())
{
    CELER_EXPECT(id_);
    CELER_EXPECT(action_reg_);
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
void ActionDiagnostic::execute(ParamsHostCRef const& params,
                               StateHostRef& state) const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    if (!(*store_))
    {
        this->build_stream_store();
    }
    MultiExceptionHandler capture_exception;
    auto launch = make_active_track_launcher(
        params,
        state,
        detail::tally_action,
        store_->state<MemSpace::host>(state.stream_id, this->num_bins()));
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
    obj["actions"] = this->actions();
    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Return the diagnostic results accumulated over all streams.
 *
 * This builds a map of particle/action combinations to the number of
 * occurances over all events.
 */
auto ActionDiagnostic::actions() const -> MapStringCount
{
    CELER_EXPECT(*store_);

    auto const& params = store_->params<MemSpace::host>();

    // Get the raw data accumulated over all host/device streams
    auto counts = this->calc_actions();

    // Map particle ID/action ID to name and store counts
    MapStringCount result;
    for (ActionId aid : range(ActionId{params.num_actions}))
    {
        for (ParticleId pid : range(ParticleId{params.num_particles}))
        {
            size_type bin = aid.get() * particle_->size() + pid.get();
            CELER_ASSERT(bin < counts.size());

            if (counts[bin] > 0)
            {
                std::string label = action_reg_->id_to_label(aid) + " "
                                    + particle_->id_to_label(pid);
                result[label] = counts[bin];
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the tallied actions accumulated over all streams.
 */
auto ActionDiagnostic::calc_actions() const -> VecCount
{
    CELER_EXPECT(*store_);

    VecCount result(this->num_bins(), 0);
    apply_to_all_streams(*store_, SumCounts{&result});
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Number of tally bins (number of particles times number of actions).
 */
size_type ActionDiagnostic::num_bins() const
{
    CELER_EXPECT(*store_);

    auto const& params = store_->params<MemSpace::host>();
    return params.num_actions * params.num_particles;
}

//---------------------------------------------------------------------------//
/*!
 * Reset diagnostic results.
 */
void ActionDiagnostic::clear()
{
    CELER_EXPECT(*store_);

    apply_to_all_streams(
        *store_, [](auto& state) { fill(size_type(0), &state.counts); });
}

//---------------------------------------------------------------------------//
/*!
 * Build the storage for diagnostic parameters and stream-dependent states.
 */
void ActionDiagnostic::build_stream_store() const
{
    CELER_EXPECT(!(*store_));

    HostVal<ActionDiagnosticParamsData> host_params;
    host_params.num_actions = action_reg_->num_actions();
    host_params.num_particles = particle_->size();
    *store_ = {std::move(host_params), num_streams_};

    CELER_ENSURE(*store_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
