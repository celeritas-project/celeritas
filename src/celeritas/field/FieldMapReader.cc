//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldMapReader.cc
//---------------------------------------------------------------------------//
#include "FieldMapReader.hh"

#include <fstream>
#include <iomanip>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the reader using an environment variable to get the volume-based
 * RZ magnetic field map data.
 */
FieldMapReader::FieldMapReader(FieldMapParameters const& params,
                               std::string file_name)
    : params_(params)
{
    file_name_ = file_name;
    CELER_EXPECT(!file_name_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Read volume-based RZ magnetic field map data.
 */
FieldMapReader::result_type FieldMapReader::operator()() const
{
    result_type result;

    // RZ map Input format
    struct FieldMapRecord
    {
        int idx_z;  //! index of z grid
        int idx_r;  //! index of r grid
        FieldMapElement value;  //! z and r components of the field in tesla
    };

    result.params = params_;

    // Store field values from the map file
    std::ifstream ifile_(file_name_,
                         std::ios::in | std::ios::binary | std::ios::ate);

    CELER_VALIDATE(ifile_,
                   << "failed to open '" << file_name_
                   << "' (should contain cross section data)");

    FieldMapRecord fd;
    std::ifstream::pos_type fsize = ifile_.tellg();
    size_type ngrid = fsize / sizeof(FieldMapRecord);
    ifile_.seekg(0, std::ios::beg);

    result.data.reserve(ngrid);

    for ([[maybe_unused]] auto i : range(ngrid))
    {
        ifile_.read(reinterpret_cast<char*>(&fd), sizeof(FieldMapRecord));
        result.data.push_back(fd.value);
    }
    ifile_.close();

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
