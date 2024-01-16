//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/GeantStepDiagnostic.cc
//---------------------------------------------------------------------------//
#include "GeantStepDiagnostic.hh"

#include <algorithm>
#include <set>
#include <G4Track.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "celeritas/ext/GeantUtils.hh"
#include "accel/ExceptionConverter.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with number of bins and threads.
 *
 * The final bin is for overflow.
 */
GeantStepDiagnostic::GeantStepDiagnostic(size_type num_bins,
                                         size_type num_threads)
    : num_bins_(num_bins + 2)
{
    CELER_EXPECT(num_bins_ > 2);
    CELER_EXPECT(num_threads > 0);

    thread_store_.resize(num_threads);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void GeantStepDiagnostic::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto obj = json::object();

    obj["steps"] = this->CalcSteps();
    obj["pdg"] = this->GetPDGs();
    obj["_index"] = {"particle", "num_steps"};

    j->obj = std::move(obj);
#else
    CELER_DISCARD(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Update the step count from the given track.
 */
void GeantStepDiagnostic::Update(G4Track const* track)
{
    CELER_EXPECT(track);

    size_type num_steps = track->GetCurrentStepNumber();
    if (num_steps == 0)
    {
        // Don't tally if the track wasn't transported with Geant4
        return;
    }

    auto thread_id = static_cast<size_type>(get_geant_thread_id());
    CELER_ASSERT(thread_id < thread_store_.size());

    // Get the vector of counts for this particle
    auto pdg = track->GetDefinition()->GetPDGEncoding();
    VecCount& counts = thread_store_[thread_id][pdg];
    counts.resize(num_bins_);

    // Increment the bin corresponding to the given step count
    size_type bin = std::min(num_steps, num_bins_ - 1);
    ++counts[bin];
}

//---------------------------------------------------------------------------//
/*!
 * Get the diagnostic results accumulated over all threads.
 */
auto GeantStepDiagnostic::CalcSteps() const -> VecVecCount
{
    auto pdgs = this->GetPDGs();
    VecVecCount result(pdgs.size(), VecCount(num_bins_));

    for (auto const& pdg_to_count : thread_store_)
    {
        for (auto pdg_idx : range(pdgs.size()))
        {
            auto iter = pdg_to_count.find(pdgs[pdg_idx]);
            if (iter != pdg_to_count.end())
            {
                for (auto step_idx : range(iter->second.size()))
                {
                    result[pdg_idx][step_idx] += iter->second[step_idx];
                }
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a sorted vector of PDGs.
 */
std::vector<int> GeantStepDiagnostic::GetPDGs() const
{
    std::set<int> pdgs;
    for (auto const& pdg_to_count : thread_store_)
    {
        for (auto const& kv : pdg_to_count)
        {
            pdgs.insert(kv.first);
        }
    }
    std::vector<int> result(pdgs.begin(), pdgs.end());
    std::sort(result.begin(), result.end());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
