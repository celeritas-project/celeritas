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
#include "celeritas/global/ActionRegistry.hh"
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
 * Construct with ID, action, and particle data.
 *
 * Since the total number of actions is needed for the diagnostic params data,
 * the construction of the stream storage is deferred until all actions have
 * been registered.
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
    detail::tally_action(
        params,
        state,
        store_->state<MemSpace::host>(state.stream_id, this->num_bins()));
}

//---------------------------------------------------------------------------//
/*!
 * Execute action with device data.
 */
void ActionDiagnostic::execute(ParamsDeviceCRef const& params,
                               StateDeviceRef& state) const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    if (!(*store_))
    {
        this->build_stream_store();
    }
    detail::tally_action(
        params,
        state,
        store_->state<MemSpace::device>(state.stream_id, this->num_bins()));
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
    obj["actions"] = this->particle_actions();
    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Return the diagnostic results accumulated over all streams.
 */
auto ActionDiagnostic::particle_actions() const -> MapStringCount
{
    CELER_EXPECT(*store_);

    MapStringCount result;

    // Get the data accumulated over all host/device streams
    auto counts = this->calc_particle_actions();

    // Map particle ID/action ID to name and store counts
    auto const& params = store_->params<MemSpace::host>();
    for (ActionId aid : range(ActionId{params.num_actions}))
    {
        for (ParticleId pid : range(ParticleId{params.num_particles}))
        {
            size_type bin = aid.get() * particle_->size() + pid.get();
            CELER_ASSERT(bin < counts.size());

            if (counts[bin] > 0)
            {
                // Accumulate the result for this action
                std::string label = action_reg_->id_to_label(aid) + " "
                                    + particle_->id_to_label(pid);
                result[label] += counts[bin];
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the tallied actions accumulated over all streams.
 */
auto ActionDiagnostic::calc_particle_actions() const -> VecCount
{
    CELER_EXPECT(*store_);

    VecCount result(this->num_bins(), 0);
    apply_to_all_streams(*store_, SumCounts{&result});
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Number of tally bins (number of particles x number of actions).
 */
size_type ActionDiagnostic::num_bins() const
{
    CELER_EXPECT(*store_);

    auto const& params = store_->params<MemSpace::host>();
    return params.num_actions * params.num_particles;
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
