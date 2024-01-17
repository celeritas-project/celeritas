//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/LivermorePEReader.cc
//---------------------------------------------------------------------------//
#include "LivermorePEReader.hh"

#include <fstream>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/io/ImportLivermorePE.hh"
#include "celeritas/io/ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the reader using the G4LEDATA environment variable to get the path
 * to the data.
 */
LivermorePEReader::LivermorePEReader()
{
    std::string const& dir = celeritas::getenv("G4LEDATA");
    CELER_VALIDATE(!dir.empty(),
                   << "environment variable G4LEDATA is not defined (needed "
                      "to locate Livermore data)");
    path_ = dir + "/livermore/phot_epics2014";
}

//---------------------------------------------------------------------------//
/*!
 * Construct the reader with the path to the directory containing the data.
 */
LivermorePEReader::LivermorePEReader(char const* path) : path_(path)
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
LivermorePEReader::result_type
LivermorePEReader::operator()(AtomicNumber atomic_number) const
{
    CELER_EXPECT(atomic_number && atomic_number < AtomicNumber{101});

    std::string z_str = std::to_string(atomic_number.unchecked_get());
    CELER_LOG(debug) << "Reading Livermore PE data for Z=" << z_str;

    result_type result;

    // Read photoelectric effect total cross section above K-shell energy but
    // below energy limit for parameterization
    {
        std::string filename = path_ + "/pe-cs-" + z_str + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile,
                       << "failed to open '" << filename
                       << "' (should contain cross section data)");

        // Set the physics vector type and the data type
        result.xs_hi.vector_type = ImportPhysicsVectorType::free;

        // Read tabulated energies and cross sections
        double energy_min = 0.;
        double energy_max = 0.;
        size_type size = 0;
        infile >> energy_min >> energy_max >> size >> size;
        result.xs_hi.x.resize(size);
        result.xs_hi.y.resize(size);
        for (size_type i = 0; i < size; ++i)
        {
            CELER_ASSERT(infile);
            infile >> result.xs_hi.x[i] >> result.xs_hi.y[i];
        }
    }

    // Read photoelectric effect total cross section below K-shell energy
    {
        std::string filename = path_ + "/pe-le-cs-" + z_str + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile,
                       << "failed to open '" << filename
                       << "' (should contain cross section data)");

        // Set the physics vector type and the data type
        result.xs_lo.vector_type = ImportPhysicsVectorType::free;

        // Check that the file is not empty
        if (!(infile.peek() == std::ifstream::traits_type::eof()))
        {
            // Read tabulated energies and cross sections
            double energy_min = 0.;
            double energy_max = 0.;
            size_type size = 0;
            infile >> energy_min >> energy_max >> size >> size;
            result.xs_lo.x.resize(size);
            result.xs_lo.y.resize(size);
            for (size_type i = 0; i < size; ++i)
            {
                CELER_ASSERT(infile);
                infile >> result.xs_lo.x[i] >> result.xs_lo.y[i];
            }
        }
    }

    // Read subshell cross section fit parameters in low energy interval
    {
        std::string filename = path_ + "/pe-low-" + z_str + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile,
                       << "failed to open '" << filename
                       << "' (should contain subshell fit parameters)");

        // Read the number of subshells and energy threshold
        constexpr size_type num_param = 6;
        size_type num_shells = 0;
        infile >> num_shells >> num_shells >> result.thresh_lo;
        result.shells.resize(num_shells);

        // Read the binding energies and fit parameters
        for (auto& shell : result.shells)
        {
            CELER_ASSERT(infile);
            infile >> shell.binding_energy;
            shell.param_lo.resize(num_param);
            for (size_type i = 0; i < num_param; ++i)
            {
                CELER_ASSERT(infile);
                infile >> shell.param_lo[i];
            }
        }
    }

    // Read subshell cross section fit parameters in high energy interval
    {
        std::string filename = path_ + "/pe-high-" + z_str + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile,
                       << "failed to open '" << filename
                       << "' (should contain subshell fit parameters)");

        // Read the number of subshells and energy threshold
        constexpr size_type num_param = 6;
        size_type num_shells = 0;
        infile >> num_shells >> num_shells >> result.thresh_hi;
        CELER_ASSERT(num_shells == result.shells.size());

        // Read the binding energies and fit parameters
        for (auto& shell : result.shells)
        {
            CELER_ASSERT(infile);
            double binding_energy;
            infile >> binding_energy;
            CELER_ASSERT(binding_energy == shell.binding_energy);
            shell.param_hi.resize(num_param);
            for (size_type i = 0; i < num_param; ++i)
            {
                CELER_ASSERT(infile);
                infile >> shell.param_hi[i];
            }
        }
    }

    // Read tabulated subshell cross sections
    {
        std::string filename = path_ + "/pe-ss-cs-" + z_str + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile,
                       << "failed to open '" << filename
                       << "' (should contain subshell cross sections)");

        for (auto& shell : result.shells)
        {
            double min_energy = 0.;
            double max_energy = 0.;
            size_type size = 0;
            size_type shell_id = 0;
            infile >> min_energy >> max_energy >> size >> shell_id;
            shell.energy.resize(size);
            shell.xs.resize(size);
            for (size_type i = 0; i < size; ++i)
            {
                CELER_ASSERT(infile);
                infile >> shell.energy[i] >> shell.xs[i];
            }
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
