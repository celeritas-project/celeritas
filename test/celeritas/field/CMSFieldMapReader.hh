//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CMSFieldMapReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "FieldMapData.hh"
#include "RZFieldInput.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Load the CMS magnetic field map.
 *
 * The CMS magnetic field map (cmsExp.mag.3_8T) is extracted from the CMS
 * software (CMSSW) and stores r-z components of magnetic field values of
 * the CMS detector at the cm-unit grid point in the range of r[0:900] and
 * z[-1600:1600]. The format of input is CMSFieldMapInput where idx_r and
 * idx_z are indices of the 2-dimensional array[idx_z][idx_r], respectively.
 * The map is used only for the purpose of a standalone simulation with the
 * CMS detector geometry and is not a part of CMSSW.
 *
 * \deprecated Use RZFieldInput and the associated JSON IO functions instead.
 */
class CMSFieldMapReader
{
    //!@{
    //! \name Type aliases
    using result_type = RZFieldInput;
    //!@}

  public:
    // Construct the reader using the path of the map file
    CMSFieldMapReader(FieldMapParameters const& params, std::string file_name);

    // Read the volume-based CMS magnetic field map
    result_type operator()() const;

  private:
    // Shared parameters for a user defined magnetic field map
    FieldMapParameters const& params_;
    // File name containing the magnetic field map
    std::string file_name_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
