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
 * Load a user-defined CMS magnetic field map.
 *
 * XXX TODO: decribe field map
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
        int                     iz;
        int                     ir;
        detail::FieldMapElement value;
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
