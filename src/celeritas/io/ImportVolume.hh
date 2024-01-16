//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportVolume.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store logical volume properties.
 */
struct ImportVolume
{
    int material_id{-1};
    std::string name;
    std::string solid_name;

    //! Whether this represents a physical volume or is just a placeholder
    explicit operator bool() const { return material_id >= 0; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
