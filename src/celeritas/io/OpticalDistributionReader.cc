//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/OpticalDistributionReader.cc
//---------------------------------------------------------------------------//
#include "OpticalDistributionReader.hh"

#include <nlohmann/json.hpp>

#include "corecel/io/Logger.hh"
#include "celeritas/optical/gen/GeneratorDataIO.json.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with input filename.
 */
OpticalDistributionReader::OpticalDistributionReader(
    std::string const& filename)
    : infile_(filename)
{
    CELER_EXPECT(!filename.empty());
    CELER_VALIDATE(infile_,
                   << "failed to open file '" << filename << "' for reading");
}

//---------------------------------------------------------------------------//
/*!
 * Read single event from the file.
 */
auto OpticalDistributionReader::operator()() -> VecDistribution
{
    CELER_EXPECT(infile_);

    VecDistribution result;
    std::string line;

    // First record contains metadata
    std::getline(infile_, line);
    while (std::getline(infile_, line))
    {
        auto j = nlohmann::json::parse(line);
        auto d = j.get<optical::GeneratorDistributionData>();
        CELER_ASSERT(d);
        result.push_back(std::move(d));
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
