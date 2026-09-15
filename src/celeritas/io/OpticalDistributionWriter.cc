//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/OpticalDistributionWriter.cc
//---------------------------------------------------------------------------//
#include "OpticalDistributionWriter.hh"

#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Version.hh"
#include "celeritas/optical/gen/GeneratorDataIO.json.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with output filename.
 */
OpticalDistributionWriter::OpticalDistributionWriter(
    std::string const& filename)
    : outfile_(filename)
{
    CELER_EXPECT(!filename.empty());
    CELER_EXPECT(outfile_);

    if (!ends_with(filename, ".jsonl"))
    {
        CELER_LOG(warning) << "JSON event writer expects a jsonl file";
    }
    CELER_LOG(info) << "Creating JSON event file at " << filename;

    // Write metadata as the first record in the file
    nlohmann::json j = {"_version", stream_to_string(celer_version())};
    outfile_ << j.dump() << '\n';
}

//---------------------------------------------------------------------------//
/*!
 * Write distribution data.
 */
void OpticalDistributionWriter::operator()(VecDistribution const& data)
{
    CELER_EXPECT(!data.empty());

    std::scoped_lock scoped_lock{write_mutex_};
    for (auto const& d : data)
    {
        CELER_ASSERT(d);
        nlohmann::json j = d;
        outfile_ << j.dump() << '\n';
    }
    outfile_.flush();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
