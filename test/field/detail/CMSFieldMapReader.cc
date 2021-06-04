//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CMSFieldMapReader.cc
//---------------------------------------------------------------------------//
#include "CMSFieldMapReader.hh"

#include <fstream>
#include <iomanip>

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct the reader using an environment variable to get the volume-based
 * CMS magnetic field map data.
 */
CMSFieldMapReader::CMSFieldMapReader()
{
    file_name_ = std::getenv("USER_FIELD_MAP");
    CELER_VALIDATE(file_name_, << "USER_FIELD_MAP is not defined");
}

//---------------------------------------------------------------------------//
/*!
 * Read the CMS volume-based magnetic field map data.
 */
CMSFieldMapReader::result_type CMSFieldMapReader::operator()() const
{
    result_type result;

    // Set field map parameters
    result.params.num_grid_r = 900 + 1;
    result.params.num_grid_z = 2 * 1600 + 1;
    result.params.offset_z   = real_type{1600};

    // Store field values from the map file
    std::ifstream ifile_(file_name_,
                         std::ios::in | std::ios::binary | std::ios::ate);

    if (ifile_.is_open())
    {
        CMSFieldMapInput        fd;
        std::ifstream::pos_type fsize = ifile_.tellg();
        size_type               ngrid = fsize / sizeof(CMSFieldMapInput);
        ifile_.seekg(0, std::ios::beg);

        result.data.reserve(ngrid);

        for (CELER_MAYBE_UNUSED auto i : range(ngrid))
        {
            ifile_.read((char*)&fd, sizeof(CMSFieldMapInput));
            result.data.push_back(fd.value);
        }
        ifile_.close();
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
