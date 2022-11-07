//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/OrangeInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map> // IWYU pragma: export
#include <vector>

#include "corecel/cont/Label.hh"
#include "orange/BoundingBox.hh"
#include "orange/Data.hh"
#include "orange/OrangeTypes.hh"
#include "orange/Translator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Compressed input for all surface definitions in a unit.
 *
 * Including the sizes of each surface is redundant but safer.
 */
struct SurfaceInput
{
    std::vector<SurfaceType> types;  //!< Surface type enums
    std::vector<real_type>   data;   //!< Compressed surface data
    std::vector<size_type>   sizes;  //!< Size of each surface's data
    std::vector<Label>       labels; //!< Surface labels

    //! Number of surfaces
    size_type size() const { return types.size(); }

    //! Whether the surface inputs are valid
    explicit operator bool() const
    {
        return types.size() == sizes.size() && labels.size() == types.size()
               && data.size() >= types.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Input definition for a single volume.
 */
struct VolumeInput
{
    using Flags = VolumeRecord::Flags;

    //! Volume label
    Label label{};

    //! Sorted list of surface IDs in this volume
    std::vector<SurfaceId> faces{};
    //! RPN region definition for this volume, using local surface index
    std::vector<logic_int> logic{};
    //! Axis-aligned bounding box (TODO: currently unused)
    BoundingBox bbox{};

    //! Special flags
    logic_int flags{0};
    //! Masking priority (2 for regular, 1 for background)
    int zorder{};

    //! Whether the volume definition is valid
    explicit operator bool() const
    {
        return !logic.empty() || (flags & Flags::implicit_vol);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Input definition for a unit.
 */
struct UnitInput
{
    struct Daughter
    {
        UniverseId  universe_id;
        Translation translation;
    };
    using MapVolumeDaughter = std::unordered_map<VolumeId, Daughter>;

    SurfaceInput             surfaces;
    std::vector<VolumeInput> volumes;
    BoundingBox              bbox; //!< Outer bounding box
    MapVolumeDaughter        daughter_map;

    // Unit metadata
    Label label;

    //! Whether the unit definition is valid
    explicit operator bool() const { return !volumes.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Construction definition for a full ORANGE geometry.
 */
struct OrangeInput
{
    std::vector<UnitInput> units;

    // TODO: array of universe types and universe ID -> offset
    // or maybe std::variant when we require C++17

    //! Whether the unit definition is valid
    explicit operator bool() const { return !units.empty(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
