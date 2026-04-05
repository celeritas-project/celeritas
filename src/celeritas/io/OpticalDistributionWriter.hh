//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/OpticalDistributionWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <fstream>
#include <string>

#include "corecel/Macros.hh"
#include "celeritas/optical/gen/GeneratorData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Export optical distribution data to JSON.
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
    std::ofstream outfile_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
