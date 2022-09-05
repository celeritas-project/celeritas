//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CMSFieldMapReader.cc
//---------------------------------------------------------------------------//
#include "CMSFieldMapReader.hh"

#include <fstream>
#include <iomanip>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct the reader using an environment variable to get the volume-based
 * CMS magnetic field map data.
 */
CMSFieldMapReader::CMSFieldMapReader(const FieldMapParameters& params,
                                     std::string               file_name)
    : params_(params)
{
    file_name_ = file_name;
    CELER_EXPECT(!file_name_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Read the CMS volume-based magnetic field map data.
 */
CMSFieldMapReader::result_type CMSFieldMapReader::operator()() const
{
    result_type result;

    result.params = params_;

    // Store field values from the map file
    std::ifstream ifile_(file_name_,
                         std::ios::in | std::ios::binary | std::ios::ate);

    CELER_VALIDATE(ifile_,
                   << "failed to open '" << file_name_
                   << "' (should contain cross section data)");

    CMSFieldMapInput        fd;
    std::ifstream::pos_type fsize = ifile_.tellg();
    size_type               ngrid = fsize / sizeof(CMSFieldMapInput);
    ifile_.seekg(0, std::ios::beg);

    result.data.reserve(ngrid);

    for (CELER_MAYBE_UNUSED auto i : range(ngrid))
    {
        ifile_.read(reinterpret_cast<char*>(&fd), sizeof(CMSFieldMapInput));
        result.data.push_back(fd.value);
    }
    ifile_.close();

    return result;
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
