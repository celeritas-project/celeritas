//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Data.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Scalar values particular to an ORANGE geometry instance.
 */
struct OrangeParamsScalars
{
    static constexpr size_type max_level{1};

    size_type max_faces{};
    size_type max_intersections{};
    size_type max_logic_depth{};

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return max_level > 0 && max_faces > 0 && max_intersections > 0;
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
    // TODO (KENO geometry): zorder

    //! Flag values (bit field)
    enum Flags : logic_int
    {
        internal_surfaces = 0x1, //!< "Complex" distance-to-boundary
        implicit_cell     = 0x2, //!< Background/exterior cell
        simple_safety     = 0x4  //!< Fast safety calculation
    };
};

//---------------------------------------------------------------------------//
/*!
 * Data for surface-to-volume connectivity.
 */
struct SurfacesRecord
{
    using RealId = OpaqueId<real_type>;

    ItemRange<SurfaceType> types;
    ItemRange<RealId>      data_offsets;

    //! Number of surfaces stored
    CELER_FUNCTION size_type size() const { return types.size(); }

    //! True if defined consistently
    explicit CELER_FUNCTION operator bool() const
    {
        return data_offsets.size() == types.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data for surface-to-volume connectivity.
 */
struct Connectivity
{
    ItemRange<VolumeId> neighbors;
};

//---------------------------------------------------------------------------//
/*!
 * Scalar data for a single "unit" of volumes defined by surfaces.
 *
 * Surfaces each have a compile-time number of real data needed to define them.
 * (These usually are the nonzero coefficients of the quadric equation.) A
 * surface ID points to an offset into the `data` field.
 */
struct SimpleUnitRecord
{
    // Surface data
    SurfacesRecord          surfaces;
    ItemRange<Connectivity> connectivity;

    // Volume data [index by VolumeId]
    ItemRange<VolumeRecord> volumes;

    // TODO: transforms
    // TODO: acceleration structure (bvh/kdtree/grid)
    // TODO: background
    bool simple_safety{};

    //! True if defined
    explicit CELER_FUNCTION operator bool() const
    {
        return surfaces && connectivity.size() == surfaces.types.size()
               && !volumes.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent data used by ORANGE implementation.
 *
 * Most data will be accessed through the invidual units, which reference data
 * in the "storage" below. The type and index for a universe ID will determine
 * the class type and data of the Tracker to instantiate. If *only* simple
 * units are present, then the \c simple_unit data structure will just be equal
 * to a range (with the total number of universes present). Use `universe_type`
 * to switch on the type of universe; then `universe_index` to index into
 * `simple_unit` or `rect_array` or ...
 */
template<Ownership W, MemSpace M>
struct OrangeParamsData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using UnivItems = Collection<T, W, M, UniverseId>;
    using RealId    = OpaqueId<real_type>;

    //// DATA ////

    // Scalar attributes
    OrangeParamsScalars scalars;

    // High-level universe definitions
    UnivItems<UniverseType> universe_type;
    UnivItems<size_type>    universe_index;
    Items<SimpleUnitRecord> simple_unit;

    // Low-level storage
    Items<SurfaceId>    surface_ids;
    Items<VolumeId>     volume_ids;
    Items<RealId>       real_ids;
    Items<logic_int>    logic_ints;
    Items<real_type>    reals;
    Items<SurfaceType>  surface_types;
    Items<Connectivity> connectivities;
    Items<VolumeRecord> volume_records;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !universe_type.empty()
               && universe_index.size() == universe_type.size()
               && ((!volume_ids.empty() && !logic_ints.empty() && !reals.empty())
                   || surface_types.empty())
               && !volume_records.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OrangeParamsData& operator=(const OrangeParamsData<W2, M2>& other)
    {
        scalars = other.scalars;

        universe_type  = other.universe_type;
        universe_index = other.universe_index;
        simple_unit    = other.simple_unit;

        surface_ids    = other.surface_ids;
        volume_ids     = other.volume_ids;
        real_ids       = other.real_ids;
        logic_ints     = other.logic_ints;
        reals          = other.reals;
        surface_types  = other.surface_types;
        connectivities = other.connectivities;
        volume_records = other.volume_records;

        CELER_ENSURE(static_cast<bool>(*this) == static_cast<bool>(other));
        return *this;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * ORANGE state data.
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
    StateItems<SurfaceId>      surf;
    StateItems<Sense>          sense;
    StateItems<BoundaryResult> boundary;

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
            && boundary.size() == pos.size()
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

        pos      = other.pos;
        dir      = other.dir;
        vol      = other.vol;
        surf     = other.surf;
        sense    = other.sense;
        boundary = other.boundary;

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
 * Resize geometry tracking states.
 */
template<MemSpace M>
inline void resize(OrangeStateData<Ownership::value, M>* data,
                   const HostCRef<OrangeParamsData>&     params,
                   size_type                             size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(size > 0);
    resize(&data->pos, size);
    resize(&data->dir, size);
    resize(&data->vol, size);
    resize(&data->surf, size);
    resize(&data->sense, size);
    resize(&data->boundary, size);

    size_type face_states = params.scalars.max_faces * size;
    resize(&data->temp_sense, face_states);

    size_type isect_states = params.scalars.max_intersections * size;
    resize(&data->temp_face, isect_states);
    resize(&data->temp_distance, isect_states);
    resize(&data->temp_isect, isect_states);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
