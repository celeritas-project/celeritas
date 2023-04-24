//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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

// Input format
struct CMSFieldMapInput
{
    int idx_z;  //! index of z grid
    int idx_r;  //! index of r grid
    FieldMapElement value;  //! z and r components of the field
};

//---------------------------------------------------------------------------//
/*!
 * Construct the reader using an environment variable to get the volume-based
 * CMS magnetic field map data.
 */
CMSFieldMapReader::CMSFieldMapReader(FieldMapParameters const& params,
                                     std::string file_name)
    : params_(params)
{
    file_name_ = file_name;
    CELER_EXPECT(!file_name_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Read the CMS volume-based magnetic field map data.
 */
auto CMSFieldMapReader::operator()() const -> result_type
{
    result_type result;

    result.num_grid_z = params_.num_grid_z;
    result.num_grid_r = params_.num_grid_r;
    result.delta_grid = params_.delta_grid;
    result.offset_z = params_.offset_z;

    // Store field values from the map file
    std::ifstream ifile_(file_name_,
                         std::ios::in | std::ios::binary | std::ios::ate);

    CELER_VALIDATE(ifile_,
                   << "failed to open '" << file_name_
                   << "' (should contain cross section data)");

    CMSFieldMapInput fd;
    std::ifstream::pos_type fsize = ifile_.tellg();
    size_type ngrid = fsize / sizeof(CMSFieldMapInput) / 2;
    ifile_.seekg(0, std::ios::beg);

    result.field_z.resize(ngrid);
    result.field_r.resize(ngrid);

    for (auto i : range(ngrid))
    {
        ifile_.read(reinterpret_cast<char*>(&fd), sizeof(CMSFieldMapInput));
        result.field_z[i] = fd.value.value_z;
        result.field_r[i] = fd.value.value_r;
    }
    ifile_.close();

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
