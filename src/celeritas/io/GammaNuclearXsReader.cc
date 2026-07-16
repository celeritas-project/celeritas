//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/GammaNuclearXsReader.cc
//---------------------------------------------------------------------------//
#include "GammaNuclearXsReader.hh"

#include <fstream>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the reader using the G4PARTICLEXSDATA environment variable to get
 * the path to the data.
 */
GammaNuclearXsReader::GammaNuclearXsReader()
{
    std::string const& dir = celeritas::getenv("G4PARTICLEXSDATA");
    CELER_VALIDATE(!dir.empty(),
                   << "environment variable G4PARTICLEXSDATA is not defined "
                      "(needed to locate gamma-nuclear cross section data)");
    path_ = dir + "/gamma";
}

//---------------------------------------------------------------------------//
/*!
 * Construct the reader with the path to the directory containing the data.
 */
GammaNuclearXsReader::GammaNuclearXsReader(char const* path) : path_(path)
{
    CELER_EXPECT(!path_.empty());
    if (path_.back() == '/')
    {
        path_.pop_back();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Read the data for the given element.
 */
GammaNuclearXsReader::result_type
GammaNuclearXsReader::operator()(AtomicNumber atomic_number) const
{
    CELER_EXPECT(atomic_number);

    std::string z_str = std::to_string(atomic_number.unchecked_get());
    CELER_LOG(debug) << "Reading gamma-nuclear xs data for Z = " << z_str;

    result_type result;

    // Read gamma-nuclear cross section data for the given atomic_number
    {
        std::string filename = path_ + "/inel" + z_str;
        std::ifstream infile(filename);
        CELER_VALIDATE(infile,
                       << "failed to open '" << filename
                       << "' (should contain cross section data)");

        // Read tabulated energies and cross sections
        double energy_min = 0.;
        double energy_max = 0.;
        size_type size = 0;
        infile >> energy_min >> energy_max >> size >> size;
        CELER_VALIDATE(
            size > 0, << "incorrect gamma-nuclear cross section size " << size);
        result.x.resize(size);
        result.y.resize(size);

        MmSqMicroXs input_xs;
        for (size_type i = 0; i < size; ++i)
        {
            CELER_ASSERT(infile);
            // Convert to the celeritas units::barn (units::BarnXs.value())
            // from clhep::mm^2 as stored in G4PARTICLEXS/gamma/inelXX data
            infile >> result.x[i] >> input_xs.value();
            result.y[i]
                = native_value_to<units::BarnXs>(native_value_from(input_xs))
                      .value();
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
