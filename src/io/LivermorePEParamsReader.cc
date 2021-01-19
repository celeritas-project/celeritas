//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEParamsReader.cc
//---------------------------------------------------------------------------//
#include "LivermorePEParamsReader.hh"

#include <fstream>
#include <sstream>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the reader using the G4LEDATA environment variable to get the path
 * to the data.
 */
LivermorePEParamsReader::LivermorePEParamsReader()
{
    const char* env_var = std::getenv("G4LEDATA");
    CELER_VALIDATE(env_var, "Environment variable G4LEDATA is not defined.");
    std::ostringstream os;
    os << env_var << "/livermore/phot_epics2014";
    path_ = os.str();
}

//---------------------------------------------------------------------------//
/*!
 * Construct the reader with the path to the directory containing the data.
 */
LivermorePEParamsReader::LivermorePEParamsReader(const char* path)
    : path_(path)
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
LivermorePEParamsReader::result_type
LivermorePEParamsReader::operator()(int atomic_number) const
{
    CELER_EXPECT(atomic_number > 0 && atomic_number < 101);

    result_type result;
    std::string Z = std::to_string(atomic_number);

    // Read photoelectric effect total cross section above K-shell energy but
    // below energy limit for parameterization
    {
        std::string   filename = path_ + "/pe-cs-" + Z + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile, "Couldn't open '" << filename << "'");

        // Set the physics vector type and the data type
        result.xs_high.vector_type = ImportPhysicsVectorType::free;

        // Read tabulated energies and cross sections
        real_type energy_min = 0.;
        real_type energy_max = 0.;
        size_type size       = 0;
        infile >> energy_min >> energy_max >> size >> size;
        result.xs_high.x.resize(size);
        result.xs_high.y.resize(size);
        for (size_type i = 0; i < size; ++i)
        {
            CELER_ASSERT(infile);
            infile >> result.xs_high.x[i] >> result.xs_high.y[i];
        }
    }

    // Read photoelectric effect total cross section below K-shell energy
    {
        std::string   filename = path_ + "/pe-le-cs-" + Z + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile, "Couldn't open '" << filename << "'");

        // Set the physics vector type and the data type
        result.xs_low.vector_type = ImportPhysicsVectorType::free;

        // Check that the file is not empty
        if (!(infile.peek() == std::ifstream::traits_type::eof()))
        {
            // Read tabulated energies and cross sections
            real_type energy_min = 0.;
            real_type energy_max = 0.;
            size_type size       = 0;
            infile >> energy_min >> energy_max >> size >> size;
            result.xs_low.x.resize(size);
            result.xs_low.y.resize(size);
            for (size_type i = 0; i < size; ++i)
            {
                CELER_ASSERT(infile);
                infile >> result.xs_low.x[i] >> result.xs_low.y[i];
            }
        }
    }

    // Read subshell cross section fit parameters in low energy interval
    {
        std::string   filename = path_ + "/pe-low-" + Z + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile, "Couldn't open '" << filename << "'");

        // Read the number of subshells and energy threshold
        constexpr size_type num_param  = 6;
        size_type           num_shells = 0;
        real_type           threshold  = 0.;
        infile >> num_shells >> num_shells >> threshold;
        result.thresh_low = units::MevEnergy{threshold};
        result.shells.resize(num_shells);

        // Read the binding energies and fit parameters
        for (auto& shell : result.shells)
        {
            CELER_ASSERT(infile);
            real_type binding_energy;
            infile >> binding_energy;
            shell.binding_energy = units::MevEnergy{binding_energy};
            shell.param_low.resize(num_param);
            for (size_type i = 0; i < num_param; ++i)
            {
                infile >> shell.param_low[i];
            }
        }
    }

    // Read subshell cross section fit parameters in high energy interval
    {
        std::string   filename = path_ + "/pe-high-" + Z + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile, "Couldn't open '" << filename << "'");

        // Read the number of subshells and energy threshold
        constexpr size_type num_param  = 6;
        size_type           num_shells = 0;
        real_type           threshold  = 0.;
        infile >> num_shells >> num_shells >> threshold;
        result.thresh_high = units::MevEnergy{threshold};
        CELER_ASSERT(num_shells == result.shells.size());

        // Read the binding energies and fit parameters
        for (auto& shell : result.shells)
        {
            CELER_ASSERT(infile);
            real_type binding_energy;
            infile >> binding_energy;
            CELER_ASSERT(binding_energy == shell.binding_energy.value());
            shell.param_high.resize(num_param);
            for (size_type i = 0; i < num_param; ++i)
            {
                infile >> shell.param_high[i];
            }
        }
    }

    // Read tabulated subshell cross sections
    {
        std::string   filename = path_ + "/pe-ss-cs-" + Z + ".dat";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile, "Couldn't open '" << filename << "'");

        for (auto& shell : result.shells)
        {
            real_type min_energy = 0.;
            real_type max_energy = 0.;
            size_type size       = 0;
            size_type shell_id   = 0;
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
} // namespace celeritas
