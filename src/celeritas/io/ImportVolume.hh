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
 * Store region description and attributes.
 */
struct ImportRegion
{
    std::string name;
    bool field_manager{false};
    bool production_cuts{false};
    bool user_limits{false};
};

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
    static inline constexpr unsigned int unspecified = -1;

    unsigned int geo_material_id{unspecified};  //!< Material defined by
                                                //!< geometry
    unsigned int region_id{unspecified};  //!< Optional region associated
    unsigned int phys_material_id{unspecified};  //!< Material modified by
                                                 //!< physics
    std::string name;
    std::string solid_name;

    //! Whether this represents a physical volume or is just a placeholder
    explicit operator bool() const { return geo_material_id != unspecified; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
