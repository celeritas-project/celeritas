//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnosticBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/data/StreamStore.hh"
#include "corecel/io/OutputInterface.hh"

#include "ParticleTallyData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Tally number of steps taken by each particle type.
 *
 * This adds an \c step-diagnostic entry to the \c result category of the
 * main Celeritas output that bins the total number of steps taken by a track,
 * grouped by particle type. The result is an integral over all events.
 */
class StepDiagnosticBase : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using StoreT = StreamStore<ParticleTallyParamsData, ParticleTallyStateData>;
    using VecVecCount = std::vector<std::vector<size_type>>;
    //!@}

  public:
    //! Construct with counts
    StepDiagnosticBase(size_type num_particles,
                       size_type max_bins,
                       size_type num_streams);

    //! Default destructor
    ~StepDiagnosticBase();

    //!@{
    //! \name Output interface

    //! Category of data to write
    Category category() const final { return Category::result; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    // Get the diagnostic results accumulated over all streams
    VecVecCount calc_steps() const;

    // Size of diagnostic state data (number of bins times number of particles)
    size_type state_size() const;

    // Access the storage
    StoreT& store() const { return store_; }

    // Reset diagnostic results
    void clear();

  private:
    size_type num_streams_;
    mutable StoreT store_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
