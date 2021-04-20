//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportVolume.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store data for each geometry volume.
 */
struct ImportVolume
{
    unsigned int volume_id;
    unsigned int material_id;
    std::string  name;
    std::string  solid_name;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
