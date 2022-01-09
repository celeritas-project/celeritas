//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "../Data.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Compressed input for all surface definitions in a universe.
 *
 * Including the sizes of each surface is redundant but safer.
 */
struct SurfaceInput
{
    std::vector<SurfaceType> types; //!< Surface type enums
    std::vector<real_type>   data;  //!< Compressed surface data
    std::vector<size_type>   sizes; //!< Size of each surface's data
};

//---------------------------------------------------------------------------//
} // namespace celeritas
