//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnosticBase.cc
//---------------------------------------------------------------------------//
#include "StepDiagnosticBase.hh"

#include <type_traits>
#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/LabelIO.json.hh"

using namespace celeritas::literals;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with counts.
 */
StepDiagnosticBase::StepDiagnosticBase(size_type num_particles,
                                       size_type max_step_bin,
                                       size_type num_streams)
    : num_streams_(num_streams)
{
    CELER_EXPECT(num_particles > 0);
    CELER_EXPECT(num_streams_ > 0);

    CELER_VALIDATE(max_step_bin > 0,
                   << "nonpositive step diagnostic 'max' bin");

    // Add two extra bins for underflow and overflow
    HostVal<ParticleTallyParamsData> host_params;
    host_params.num_bins = max_step_bin + 2;
    host_params.num_particles = num_particles;
    store_ = {std::move(host_params), num_streams_};

    CELER_ENSURE(store_);
}

//---------------------------------------------------------------------------//
//! Default destructor
StepDiagnosticBase::~StepDiagnosticBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void StepDiagnosticBase::output(JsonPimpl* j) const
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
auto StepDiagnosticBase::calc_steps() const -> VecVecCount
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
size_type StepDiagnosticBase::state_size() const
{
    auto const& params = store_.params<MemSpace::host>();
    return params.num_bins * params.num_particles;
}

//---------------------------------------------------------------------------//
/*!
 * Reset diagnostic results.
 */
void StepDiagnosticBase::clear()
{
    apply_to_all_streams(store_,
                         [](auto& state) { fill(0_sz, &state.counts); });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
