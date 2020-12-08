//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportVolume.hh
//! Store volume information.
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store data for each volume.
 *
 * Used by the GdmlGeometryMap class.
 *
 * The reason for this (currently absurdly simple) struct instead of just
 * keeping tab of volume and solid names directly in the GdmlGeometryMap, is
 * to easily import more information in the future.
 */
struct ImportVolume
{
    std::string name;
    std::string solid_name;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
