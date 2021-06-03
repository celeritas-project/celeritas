//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CMSFieldMapReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include "FieldMapInterface.hh"

namespace celeritas
{
namespace detail
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
 */
class CMSFieldMapReader
{
    //!@{
    //! Type aliases
    using result_type = detail::FieldMapData;
    //!@}

    // Input format
    struct CMSFieldMapInput
    {
        int                     idx_z; //! [-1600:1600]
        int                     idx_r; //! [0:900]
        detail::FieldMapElement value; //! z and r components of the field
    };

  public:
    // Construct the reader using the environment variable
    CMSFieldMapReader();

    // Read the volume-based CMS magnetic field map
    result_type operator()() const;

  private:
    // File name containing the magnetic field map
    char* file_name_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
