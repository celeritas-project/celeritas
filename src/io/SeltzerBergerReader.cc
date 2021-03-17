//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBergerReader.cc
//---------------------------------------------------------------------------//
#include "SeltzerBergerReader.hh"

#include <fstream>
#include <sstream>
#include "base/Assert.hh"
#include "base/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct using environmental variable $G4LEDATA.
 */
SeltzerBergerReader::SeltzerBergerReader()
{
    const char* env_var = std::getenv("G4LEDATA");
    CELER_VALIDATE(env_var, "Environment variable G4LEDATA is not defined.");
    std::ostringstream os;
    os << env_var << "/brem_SB";
    path_ = os.str();
}

//---------------------------------------------------------------------------//
/*!
 * Construct using a user defined path to the folder containing the data.
 * The path should point to the files that are usually stored in
 * [Geant4-install]/share/Geant4-10.7.0/data/G4EMLOW7.12/brem_SB/.
 */
SeltzerBergerReader::SeltzerBergerReader(const char* path) : path_(path)
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
    CELER_EXPECT(atomic_number > 0);

    // For Z = 93-99, the incident log energy grid and reduced photon energy
    // grid in the bremsstrahlung data files are incorrect. These are the grids
    // that should be used for those elements (and for all Z < 100).
    constexpr double log_energy[]
        = {-6.9078,  -6.5023,  -6.2146,  -5.8091, -5.5215, -5.2983, -5.116,
           -4.8283,  -4.6052,  -4.1997,  -3.912,  -3.5066, -3.2189, -2.9957,
           -2.8134,  -2.5257,  -2.3026,  -1.8971, -1.6094, -1.204,  -0.91629,
           -0.69315, -0.51083, -0.22314, 0,       0.40547, 0.69315, 1.0986,
           1.3863,   1.6094,   1.7918,   2.0794,  2.3026,  2.7081,  2.9957,
           3.4012,   3.6889,   3.912,    4.0943,  4.382,   4.6052,  5.0106,
           5.2983,   5.7038,   5.9915,   6.2146,  6.3969,  6.6846,  6.9078,
           7.3132,   7.6009,   8.0064,   8.294,   8.5172,  8.6995,  8.9872,
           9.2103};
    constexpr double kappa[]
        = {1e-12, 0.025, 0.05,  0.075,  0.1,    0.15,    0.2,     0.25,
           0.3,   0.35,  0.4,   0.45,   0.5,    0.55,    0.6,     0.65,
           0.7,   0.75,  0.8,   0.85,   0.9,    0.925,   0.95,    0.97,
           0.99,  0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 1.};

    result_type result;

    // Open file for given atomic number
    std::string   file = path_ + "/br" + std::to_string(atomic_number);
    std::ifstream input_stream(file.c_str());
    CELER_VALIDATE(input_stream, "Could not open file " << file << ".");

    // Fetch binning information
    unsigned int g4_physics_vector_type; // Not used
    unsigned int x_size = 0;
    unsigned int y_size = 0;

    input_stream >> g4_physics_vector_type >> y_size >> x_size;
    result.x.resize(x_size);
    result.y.resize(y_size);

    // Read reduced photon energy grid
    for (auto i : range(y_size))
    {
        CELER_ASSERT(input_stream);
        input_stream >> result.y[i];
    }

    // Read incident particle log energy grid
    for (auto i : range(x_size))
    {
        CELER_ASSERT(input_stream);
        input_stream >> result.x[i];
    }

    // Correct the energy grids for Z = 93-99
    if (atomic_number > 92 && atomic_number < 100)
    {
        result.x = std::vector<double>(std::begin(log_energy),
                                       std::end(log_energy));
        result.y = std::vector<double>(std::begin(kappa), std::end(kappa));
        x_size   = result.x.size();
        y_size   = result.y.size();
    }

    // Read scaled differential cross sections, storing in row-major order
    result.value.resize(x_size * y_size);
    for (auto i : range(x_size))
    {
        for (auto j : range(y_size))
        {
            CELER_ASSERT(input_stream);
            input_stream >> result.value[i * y_size + j];
        }
    }

    CELER_ENSURE(!result.x.empty() && !result.y.empty());
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
