//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * "geo material ID" is the index of the G4Material (physical properties).
 * (TODO: these should be indexed using a future GeantModel class, not using
 * the internal GetIndex method).
 *
 * \note The index of this volume in the \c volumes vector is the canonical
 * \c VolumeId created by the Geant4 geometry model.
 */
struct ImportVolume
{
    //!@{
    //! \name Type aliases
    using Index = unsigned int;
    //!@}

    static constexpr Index unspecified = -1;

    Index geo_material_id{unspecified};  //!< Material defined by geometry
    Index region_id{unspecified};  //!< Optional region associated
    Index phys_material_id{unspecified};  //!< Material modified by physics
    std::string name;
    std::string solid_name;

    //! Whether this represents a physical volume or is just a placeholder
    explicit operator bool() const { return geo_material_id != unspecified; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
