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
 *
 * \note The "phys material ID" is the index of the MaterialCutsCouple, and the
 * "geo material ID" is the index of the Material (physical properties).
 *
 * \note The index of this volume in the \c volumes vector is the "instance ID"
 * which is not necessarily reproducible.
 */
struct ImportVolume
{
    unsigned int geo_material_id{};  //!< Actual material properties
    unsigned int material_id{};  //!< Material properties modified by physics
    std::string name;
    std::string solid_name;

    //! Whether this represents a physical volume or is just a placeholder
    explicit operator bool() const { return !name.empty(); }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
