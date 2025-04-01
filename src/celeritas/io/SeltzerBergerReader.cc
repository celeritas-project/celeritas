//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/SeltzerBergerReader.cc
//---------------------------------------------------------------------------//
#include "SeltzerBergerReader.hh"

#include <fstream>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct using environmental variable $G4LEDATA.
 */
SeltzerBergerReader::SeltzerBergerReader()
{
    std::string const& dir = celeritas::getenv("G4LEDATA");
    CELER_VALIDATE(!dir.empty(),
                   << "environment variable G4LEDATA is not defined (needed "
                      "to locate Seltzer-Berger data)");
    path_ = dir + "/brem_SB";
}

//---------------------------------------------------------------------------//
/*!
 * Construct using a user defined path to the folder containing the data.
 * The path should point to the files that are usually stored in
 * [Geant4-install]/share/Geant4-10.7.0/data/G4EMLOW7.12/brem_SB/.
 */
SeltzerBergerReader::SeltzerBergerReader(char const* path) : path_(path)
{
    CELER_EXPECT(path_.size());
    if (path_.back() == '/')
    {
        path_.pop_back();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Fetch data for a given atomic number.
 *
 * Standard data files encompass Z = [1, 100].
 */
SeltzerBergerReader::result_type
SeltzerBergerReader::operator()(AtomicNumber atomic_number) const
{
    CELER_EXPECT(atomic_number);

    std::string z_str = std::to_string(atomic_number.unchecked_get());
    result_type result;

    // Open file for given atomic number
    std::string file = path_ + "/br" + z_str;
    std::ifstream input_stream(file.c_str());
    CELER_VALIDATE(input_stream,
                   << "failed to open '" << file
                   << "' (should contain SB cross section data)");

    // Fetch binning information
    unsigned int dummy;
    unsigned int x_size = 0;
    unsigned int y_size = 0;

    input_stream >> dummy >> y_size >> x_size;
    CELER_ASSERT(input_stream);
    result.x.resize(x_size);
    result.y.resize(y_size);

    // Read reduced photon energy grid
    for (auto i : range(y_size))
    {
        input_stream >> result.y[i];
        CELER_ASSERT(input_stream);
    }

    // Read incident particle log energy grid
    for (auto i : range(x_size))
    {
        input_stream >> result.x[i];
        CELER_ASSERT(input_stream);
    }

    // Read scaled differential cross sections, storing in row-major order
    result.value.resize(x_size * y_size);
    for (auto i : range(x_size))
    {
        for (auto j : range(y_size))
        {
            input_stream >> result.value[i * y_size + j];
            CELER_ASSERT(input_stream);
        }
    }

    // Check that we've reached the end of the file
    input_stream >> dummy;
    CELER_VALIDATE(!input_stream,
                   << "unexpected end of file '" << file
                   << "' (data is inconsistent with boundaries)");

    CELER_ENSURE(!result.x.empty() && !result.y.empty());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
