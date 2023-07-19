//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/TrackStepCounter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <G4Track.hh>

#include "corecel/Types.hh"
#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Tally the steps per track transported with Geant4 for each particle type.
 *
 * For the diagnostic class that collects the same result for tracks
 * transported with Celeritas, see: \sa StepDiagnostic.
 */
class TrackStepCounter final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using VecCount = std::vector<size_type>;
    using VecVecCount = std::vector<VecCount>;
    using MapIntVecCount = std::unordered_map<int, VecCount>;
    //!@}

  public:
    // Construct in an uninitialized state
    TrackStepCounter() = default;

    // Construct with number of bins and threads
    TrackStepCounter(size_type num_bins, size_type num_threads);

    //! Default destructor
    ~TrackStepCounter();

    //!@{
    //! \name Output interface
    //! Category of data to write
    Category category() const final { return Category::result; }
    //! Key for the entry inside the category.
    std::string label() const final { return "track-step-count"; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    // Initialize the diagnostic on the "master" thread
    inline void Initialize(size_type num_bins, size_type num_threads);

    // Update the step tally from the given track
    void Update(G4Track const* track);

    // Get the results accumulated over all threads
    VecVecCount CalcSteps() const;

    // Get a sorted vector of PDGs
    std::vector<int> GetPDGs() const;

    //! Whether this instance is initialized
    explicit operator bool() const { return !thread_store_.empty(); }

  private:
    std::vector<MapIntVecCount> thread_store_;
    size_type num_bins_;
};

//---------------------------------------------------------------------------//
/*!
 * Initialize the diagnostic on the "master" thread.
 */
void TrackStepCounter::Initialize(size_type num_bins, size_type num_threads)
{
    *this = TrackStepCounter(num_bins, num_threads);
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
