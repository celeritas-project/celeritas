//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Data.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/CollectionBuilder.hh"
#include "base/OpaqueId.hh"
#include "geometry/Types.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Data for type-deleted surface definitions.
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

    //! Number of surfaces
    CELER_FUNCTION SurfaceId::size_type size() const { return types.size(); }

    //! True if sizes are valid
    explicit CELER_FUNCTION operator bool() const
    {
        return offsets.size() == types.size() && reals.size() >= types.size();
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
 * \sa VolumeView
 */
struct VolumeRecord
{
    ItemRange<SurfaceId> faces;
    ItemRange<logic_int> logic;

    logic_int max_intersections{0};
    logic_int flags{0};

    //! Flag values (bit field)
    enum Flags : logic_int
    {
        internal_surfaces = 0x1
    };
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

    Items<VolumeRecord> defs;

    // Storage
    Collection<SurfaceId, W, M> faces;
    Collection<logic_int, W, M> logic;

    //// METHODS ////

    //! Number of volumes
    CELER_FUNCTION VolumeId::size_type size() const { return defs.size(); }

    //! True if sizes are valid
    explicit CELER_FUNCTION operator bool() const { return !defs.empty(); }

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
 * Scalar values particular to an ORANGE geometry instance.
 */
struct OrangeParamsScalars
{
    static constexpr size_type max_level{1};
    size_type                  max_faces{};
    size_type                  max_intersections{};

    // TODO: fuzziness/length scale

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return max_level > 0 && max_faces > 0 && max_intersections > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data to persistent data used by ORANGE implementation.
 */
template<Ownership W, MemSpace M>
struct OrangeParamsData
{
    //// DATA ////

    SurfaceData<W, M> surfaces;
    VolumeData<W, M>  volumes;

    OrangeParamsScalars scalars;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return surfaces && volumes && scalars;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OrangeParamsData& operator=(const OrangeParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        surfaces = other.surfaces;
        volumes  = other.volumes;
        scalars  = other.scalars;
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
    //// TYPES ////

    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    template<class T>
    using Items = celeritas::Collection<T, W, M>;

    //// DATA ////

    // For each track, one per max_level
    StateItems<Real3>    pos;
    StateItems<Real3>    dir;
    StateItems<VolumeId> vol;

    // Surface crossing
    StateItems<SurfaceId> surf;
    StateItems<Sense>     sense;

    // Scratch space
    Items<Sense>     temp_sense;    // [track][max_faces]
    Items<FaceId>    temp_face;     // [track][max_intersections]
    Items<real_type> temp_distance; // [track][max_intersections]
    Items<size_type> temp_isect;    // [track][max_intersections]

    //// METHODS ////

    //! True if sizes are consistent and nonzero
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return !pos.empty()
            && dir.size() == pos.size()
            && vol.size() == pos.size()
            && surf.size() == pos.size()
            && sense.size() == pos.size()
            && !temp_sense.empty()
            && !temp_face.empty()
            && temp_distance.size() == temp_face.size()
            && temp_isect.size() == temp_face.size();
        // clang-format on
    }

    //! State size
    CELER_FUNCTION ThreadId::size_type size() const { return pos.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OrangeStateData& operator=(OrangeStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        pos   = other.pos;
        dir   = other.dir;
        vol   = other.vol;
        surf  = other.surf;
        sense = other.sense;

        temp_sense    = other.temp_sense;
        temp_face     = other.temp_face;
        temp_distance = other.temp_distance;
        temp_isect    = other.temp_isect;

        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize particle states in host code.
 */
template<MemSpace M>
inline void
resize(OrangeStateData<Ownership::value, M>* data,
       const OrangeParamsData<Ownership::const_reference, MemSpace::host>& params,
       size_type size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(size > 0);
    make_builder(&data->pos).resize(size);
    make_builder(&data->dir).resize(size);
    make_builder(&data->vol).resize(size);
    make_builder(&data->surf).resize(size);
    make_builder(&data->sense).resize(size);

    size_type face_states = params.scalars.max_faces * size;
    make_builder(&data->temp_sense).resize(face_states);

    size_type isect_states = params.scalars.max_intersections * size;
    make_builder(&data->temp_face).resize(isect_states);
    make_builder(&data->temp_distance).resize(isect_states);
    make_builder(&data->temp_isect).resize(isect_states);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
