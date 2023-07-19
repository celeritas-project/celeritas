//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/TrackStepCounter.cc
//---------------------------------------------------------------------------//
#include "TrackStepCounter.hh"

#include <algorithm>
#include <G4Track.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "accel/ExceptionConverter.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with number of bins and threads.
 *
 * The two extra bins are for underflow and overflow.
 */
TrackStepCounter::TrackStepCounter(size_type num_bins, size_type num_threads)
    : num_bins_(num_bins + 2)
{
    CELER_EXPECT(num_bins_ > 2);
    CELER_EXPECT(num_threads > 0);

    thread_store_.resize(num_threads);
}

//---------------------------------------------------------------------------//
//! Default destructor
TrackStepCounter::~TrackStepCounter() = default;

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void TrackStepCounter::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto obj = json::object();

    obj["steps"] = this->CalcSteps();
    obj["pdg"] = this->GetPDGs();
    obj["_index"] = {"particle", "num_steps"};

    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Update the step tally from the given track.
 */
void TrackStepCounter::Update(G4Track const* track)
{
    CELER_EXPECT(track);

    // Don't tally if the track wasn't transported with Geant4
    size_type num_steps = track->GetCurrentStepNumber();
    if (num_steps == 0)
    {
        return;
    }

    size_type thread_id = G4Threading::IsMultithreadedApplication()
                              ? G4Threading::G4GetThreadId()
                              : 0;
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
auto TrackStepCounter::CalcSteps() const -> VecVecCount
{
    auto pdgs = this->GetPDGs();
    VecVecCount result(pdgs.size(), VecCount(num_bins_));

    for (auto const& pdg_to_count : thread_store_)
    {
        for (auto pdg : range(pdgs.size()))
        {
            auto iter = pdg_to_count.find(pdgs[pdg]);
            if (iter != pdg_to_count.end())
            {
                for (auto bin : range(iter->second.size()))
                {
                    result[pdg][bin] += iter->second[bin];
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
std::vector<int> TrackStepCounter::GetPDGs() const
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
}  // namespace app
}  // namespace celeritas
