//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportVolume.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store volume data.
 *
 * \sa ImportData
 */
struct ImportVolume
{
    unsigned int material_id;
    std::string  name;
    std::string  solid_name;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
