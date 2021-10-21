//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Data.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/CollectionBuilder.hh"
#include "base/OpaqueId.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Data for surface definitions.
 *
 * Surfaces each have a compile-time number of real data needed to define them.
 * (These usually are the nonzero coefficients of the quadric equation.) A
 * surface ID points to an offset into the `data` field. These surface IDs are
 * *global* over all universes.
 */
template<Ownership W, MemSpace M>
struct SurfaceData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M, SurfaceId>;

    //// DATA ////

    Items<SurfaceType>          types;
    Items<OpaqueId<real_type>>  offsets;
    Collection<real_type, W, M> reals;

    //// METHODS ////

    //! True if sizes are valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !types.empty() && offsets.size() == types.size()
               && reals.size() >= types.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfaceData& operator=(const SurfaceData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        types   = other.types;
        offsets = other.offsets;
        reals   = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data for a single volume definition.
 *
 * A volume is a CSG tree of surfaces. Each surface defines an "inside" space
 * and "outside" space that correspond to "negative" and "positive" values of
 * the quadric expression's evaluation. Left of a plane is negative, for
 * example, and evaluates to "false" or "inside" or "negative". The CSG tree is
 * flattened into a
 */
struct VolumeDef
{
    ItemRange<SurfaceId> faces;
    ItemRange<logic_int> logic;

    logic_int num_intersections{0};
    logic_int flags{0};
};

//---------------------------------------------------------------------------//
/*!
 * Data for volume definitions.
 */
template<Ownership W, MemSpace M>
struct VolumeData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M, VolumeId>;

    //// DATA ////

    Items<VolumeDef> defs;

    // Storage
    Collection<SurfaceId, W, M> faces;
    Collection<logic_int, W, M> logic;

    //// METHODS ////

    //! True if sizes are valid
    explicit CELER_FUNCTION operator bool() const { return defs.size() > 0; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    VolumeData& operator=(const VolumeData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        defs  = other.defs;
        faces = other.faces;
        logic = other.logic;

        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data for universe definitions.
 */
template<Ownership W, MemSpace M>
struct UniverseData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M, UniverseId>;

    //// DATA ////

    //// METHODS ////

    //! True if sizes are valid
    explicit CELER_FUNCTION operator bool() const { return false; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    UniverseData& operator=(const UniverseData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Scalar values particular to an ORANGE geometry instance.
 */
struct OrangeParamsScalars
{
    size_type max_level{};
    size_type max_faces{};
    size_type max_intersections{};

    // TODO: fuzziness/length scale
};

//---------------------------------------------------------------------------//
/*!
 * Data to persistent data used by ORANGE implementation.
 */
template<Ownership W, MemSpace M>
struct OrangeParamsData
{
    //// DATA ////

    SurfaceData<W, M>  surfaces;
    VolumeData<W, M>   volumes;
    UniverseData<W, M> universes;

    OrangeParamsScalars scalars;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return surfaces && volumes && universes;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OrangeParamsData& operator=(const OrangeParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        surfaces  = other.surfaces;
        volumes   = other.volumes;
        universes = other.universes;
        scalars   = other.scalars;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Data to persistent data used by ORANGE implementation.
 */
template<Ownership W, MemSpace M>
struct OrangeStateData
{
};

//---------------------------------------------------------------------------//
} // namespace celeritas
