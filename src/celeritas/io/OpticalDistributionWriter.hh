//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/OpticalDistributionWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <fstream>
#include <mutex>
#include <string>

#include "corecel/Macros.hh"
#include "celeritas/optical/gen/GeneratorData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Export optical distribution data to JSONL.
 *
 * This class is thread-safe for concurrent writes: calls to \c operator() are
 * serialized using an internal mutex. The writer must be constructed on the
 * main thread.
 */
class OpticalDistributionWriter
{
  public:
    //!@{
    //! \name Type aliases
    using VecDistribution = std::vector<optical::GeneratorDistributionData>;
    //!@}

    // Construct with output filename
    explicit OpticalDistributionWriter(std::string const& filename);

    //! Prevent copying and moving
    CELER_DELETE_COPY_MOVE(OpticalDistributionWriter);
    ~OpticalDistributionWriter() = default;

    // Export primaries to json
    void operator()(VecDistribution const&);

  private:
    std::mutex write_mutex_;
    std::ofstream outfile_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
