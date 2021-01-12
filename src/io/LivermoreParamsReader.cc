//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermoreParamsReader.cc
//---------------------------------------------------------------------------//
#include "LivermoreParamsReader.hh"

#include <fstream>
#include <sstream>
#include "comm/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the reader, optionally supplying the path to the directory
 * containing the data. If no path is specified, the environment variable
 * G4LEDATA is used to locate the data.
 */
LivermoreParamsReader::LivermoreParamsReader(const char* path)
{
    if (path)
    {
        std::ostringstream os;
        os << path << "/";
        path_ = os.str();
    }
    else
    {
        const char* env_var = std::getenv("G4LEDATA");
        if (!env_var)
        {
            CELER_LOG(error) << "Environment variable G4LEDATA is not defined";
        }
        std::ostringstream os;
        os << env_var << "/livermore/phot_epics2014/";
        path_ = os.str();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Read the data for the given elements.
 */
LivermoreParamsReader::result_type
LivermoreParamsReader::operator()(ElementDefId el_id, int atomic_number)
{
    REQUIRE(atomic_number > 0 && atomic_number < 101);

    constexpr real_type barn = 1.e-24 * units::centimeter * units::centimeter;

    result_type result;
    result.el_id = el_id;

    // Read photoelectric effect total cross section above K-shell energy but
    // below energy limit for parameterization
    std::string Z        = std::to_string(atomic_number);
    std::string filename = path_ + "pe-cs-" + Z + ".dat";
    if (file_exists(filename))
    {
        std::ifstream infile(filename);
        CHECK(infile.is_open());

        // Set the physics vector type and the data type
        result.xs_high.vector_type = ImportPhysicsVectorType::free;

        // Read tabulated energies and cross sections
        real_type energy_min, energy_max;
        size_type size;
        infile >> energy_min >> energy_max >> size >> size;
        result.xs_high.x.resize(size);
        result.xs_high.y.resize(size);
        for (size_type i = 0; i < size; ++i)
        {
            infile >> result.xs_high.x[i] >> result.xs_high.y[i];
            result.xs_high.y[i] *= barn;
        }
        infile.close();
    }

    // Read photoelectric effect total cross section below K-shell energy
    filename = path_ + "pe-le-cs-" + Z + ".dat";
    if (file_exists(filename))
    {
        std::ifstream infile(filename);
        CHECK(infile.is_open());

        // Set the physics vector type and the data type
        result.xs_low.vector_type = ImportPhysicsVectorType::free;

        // Check that the file is not empty
        if (!(infile.peek() == std::ifstream::traits_type::eof()))
        {
            // Read tabulated energies and cross sections
            real_type energy_min, energy_max;
            size_type size;
            infile >> energy_min >> energy_max >> size >> size;
            result.xs_low.x.resize(size);
            result.xs_low.y.resize(size);
            for (size_type i = 0; i < size; ++i)
            {
                infile >> result.xs_low.x[i] >> result.xs_low.y[i];
                result.xs_low.y[i] *= barn;
            }
        }
        infile.close();
    }

    // Read subshell cross section fit parameters in low energy interval
    filename = path_ + "pe-low-" + Z + ".dat";
    if (file_exists(filename))
    {
        std::ifstream infile(filename);
        CHECK(infile.is_open());

        // Read the number of subshells and energy threshold
        size_type num_shells;
        real_type threshold;
        infile >> num_shells >> num_shells >> threshold;
        result.thresh_low = units::MevEnergy{threshold};
        result.shells.resize(num_shells);

        // Read the binding energies and fit parameters
        for (auto& shell : result.shells)
        {
            real_type binding_energy;
            infile >> binding_energy;
            shell.binding_energy = units::MevEnergy{binding_energy};
            shell.param_low.resize(6);
            for (size_type i = 0; i < shell.param_low.size(); ++i)
            {
                infile >> shell.param_low[i];
            }
        }
        infile.close();
    }

    // Read subshell cross section fit parameters in high energy interval
    filename = path_ + "pe-high-" + Z + ".dat";
    if (file_exists(filename))
    {
        std::ifstream infile(filename);
        CHECK(infile.is_open());

        // Read the number of subshells and energy threshold
        size_type num_shells;
        real_type threshold;
        infile >> num_shells >> num_shells >> threshold;
        result.thresh_high = units::MevEnergy{threshold};
        CHECK(num_shells == result.shells.size());

        // Read the binding energies and fit parameters
        for (auto& shell : result.shells)
        {
            real_type binding_energy;
            infile >> binding_energy;
            CHECK(binding_energy == shell.binding_energy.value());
            shell.param_high.resize(6);
            for (size_type i = 0; i < shell.param_high.size(); ++i)
            {
                infile >> shell.param_high[i];
            }
        }
        infile.close();
    }

    // Read tabulated subshell cross sections
    filename = path_ + "pe-ss-cs-" + Z + ".dat";
    if (file_exists(filename))
    {
        std::ifstream infile(filename);
        CHECK(infile.is_open());

        for (auto& shell : result.shells)
        {
            real_type min_energy, max_energy;
            size_type size, shell_id;
            infile >> min_energy >> max_energy >> size >> shell_id;
            shell.energy.resize(size);
            shell.xs.resize(size);
            for (size_type i = 0; i < size; ++i)
            {
                infile >> shell.energy[i] >> shell.xs[i];
                shell.xs[i] *= barn;
            }
        }
        infile.close();
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Check that the file exists, and raise an error if not.
 */
bool LivermoreParamsReader::file_exists(const std::string& filename)
{
    bool result = std::ifstream(filename).good();
    if (!result)
    {
        CELER_LOG(error) << "File " << filename << " does not exist";
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
