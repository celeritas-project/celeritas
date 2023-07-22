//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/GeantStepDiagnostic.hh
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
class GeantStepDiagnostic final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using VecCount = std::vector<size_type>;
    using VecVecCount = std::vector<VecCount>;
    using MapIntVecCount = std::unordered_map<int, VecCount>;
    //!@}

  public:
    // Construct with number of bins and threads
    GeantStepDiagnostic(size_type num_bins, size_type num_threads);

    //! Default destructor
    ~GeantStepDiagnostic();

    //!@{
    //! \name Output interface
    //! Category of data to write
    Category category() const final { return Category::result; }
    //! Key for the entry inside the category.
    std::string label() const final { return "g4-step-diagnostic"; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    // Update the step tally from the given track
    void Update(G4Track const* track);

    // Get the results accumulated over all threads
    VecVecCount CalcSteps() const;

    // Get a sorted vector of PDGs
    std::vector<int> GetPDGs() const;

  private:
    std::vector<MapIntVecCount> thread_store_;
    size_type num_bins_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
