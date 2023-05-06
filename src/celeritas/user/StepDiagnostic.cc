//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.cc
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "celeritas/phys/ParticleParams.hh"
+ #include "celeritas/global/CoreParams.hh"

#include "detail/StepDiagnosticImpl.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "corecel/io/LabelIO.json.hh"
#endif

    namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with particle data.
 */
StepDiagnostic::StepDiagnostic(ActionId id,
                               SPConstParticle particle,
                               size_type max_steps,
                               size_type num_streams)
    : id_(id), num_streams_(num_streams)
{
    CELER_EXPECT(id_);
    CELER_EXPECT(particle);
    CELER_EXPECT(max_steps > 0);
    CELER_EXPECT(num_streams_ > 0);

    // Add two extra bins for underflow and overflow
    HostVal<ParticleTallyParamsData> host_params;
    host_params.num_bins = max_steps + 2;
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
void StepDiagnostic::execute(CoreParams const& params, StateHostRef& state)
    const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    MultiExceptionHandler capture_exception;
    auto launch = make_active_track_launcher(
        params.ref<MemSpace::native>(),
        state,
        detail::tally_steps,
        store_.params<MemSpace::host>(),
        store_.state<MemSpace::host>(state.stream_id, this->state_size()));
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
void StepDiagnostic::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto obj = json::object();

    obj["steps"] = this->calc_steps();
    obj["_index"] = {"particle", "num_steps"};

    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
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
