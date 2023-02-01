//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"

#include "OrangeTypes.hh"

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
    static constexpr size_type max_level{20};

    size_type max_faces{};
    size_type max_intersections{};
    size_type max_logic_depth{};

    // Multiplicative and additive values for bumping particles
    real_type bump_rel{1e-8};
    real_type bump_abs{1e-8};

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return max_level > 0 && max_faces > 0 && max_intersections > 0
               && bump_rel > 0 && bump_abs > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data for a single volume definition.
 *
 * Surface IDs are local to the unit.
 *
 * \sa VolumeView
 */
struct VolumeRecord
{
    ItemRange<SurfaceId> faces;
    ItemRange<logic_int> logic;

    logic_int max_intersections{0};
    logic_int flags{0};
    UniverseId daughter;
    TranslationId daughter_translation;
    // TODO (KENO geometry): zorder

    //! Flag values (bit field)
    enum Flags : logic_int
    {
        internal_surfaces = 0x1,  //!< "Complex" distance-to-boundary
        implicit_vol = 0x2,  //!< Background/exterior volume
        simple_safety = 0x4,  //!< Fast safety calculation
        embedded_universe = 0x8  //!< Volume contains embeddded universe
    };
};

//---------------------------------------------------------------------------//
/*!
 * Data for surfaces within a single unit.
 *
 * Surfaces each have a compile-time number of real data needed to define them.
 * (These usually are the nonzero coefficients of the quadric equation.) The
 * two fields in this record point to the collapsed surface types and
 * linearized data for all surfaces in a unit.
 *
 * The "types" and "data offsets" are both indexed into using the local surface
 * ID. The result of accessing "data offset" is an index into the \c real_ids
 * array, which then points us to the start address in \c reals. This marks the
 * beginning of the data used by the surface. Since the surface type tells us
 * the number of real values needed for that surface, we implicitly get a Span
 * of real values with a single indirection.
 */
struct SurfacesRecord
{
    using RealId = OpaqueId<real_type>;

    ItemRange<SurfaceType> types;
    ItemRange<RealId> data_offsets;

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
 *
 * This struct is associated with a specific surface; the \c neighbors range is
 * a list of local volume IDs for that surface.
 */
struct Connectivity
{
    ItemRange<VolumeId> neighbors;
};

//---------------------------------------------------------------------------//
/*!
 * Scalar data for a single "unit" of volumes defined by surfaces.
 */
struct SimpleUnitRecord
{
    using VolumeRecordRange = Range<VolumeId>;

    // Surface data
    SurfacesRecord surfaces;
    ItemRange<Connectivity> connectivity;  // Index by SurfaceId

    // Volume data [index by VolumeId]
    VolumeRecordRange volumes;

    // Translation data [index by TranslationId]
    ItemRange<Translation> translations;

    // TODO: transforms
    // TODO: acceleration structure (bvh/kdtree/grid)
    VolumeId background{};  //!< Default if not in any other volume
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
 * Surface and volume offsets to convert between local and global indices.
 *
 * Each collection should be of length num_universes + 1. The first entry is
 * zero and the last item should be the total number of surfaces or volumes.
 */
template<Ownership W, MemSpace M>
struct UnitIndexerData
{
    Collection<size_type, W, M> surfaces;
    Collection<size_type, W, M> volumes;

    template<Ownership W2, MemSpace M2>
    UnitIndexerData& operator=(UnitIndexerData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        surfaces = other.surfaces;
        volumes = other.volumes;

        CELER_ENSURE(*this);
        return *this;
    }

    explicit CELER_FUNCTION operator bool() const
    {
        return !surfaces.empty() && !volumes.empty();
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
    template<class T>
    using VolumeItems = Collection<T, W, M, VolumeId>;
    using RealId = OpaqueId<real_type>;

    //// DATA ////

    // Scalar attributes
    OrangeParamsScalars scalars;

    // High-level universe definitions
    UnivItems<UniverseType> universe_type;
    UnivItems<size_type> universe_index;
    Items<SimpleUnitRecord> simple_unit;

    // Low-level storage
    Items<SurfaceId> surface_ids;
    Items<VolumeId> volume_ids;
    Items<RealId> real_ids;
    Items<logic_int> logic_ints;
    Items<real_type> reals;
    Items<SurfaceType> surface_types;
    Items<Connectivity> connectivities;
    VolumeItems<VolumeRecord> volume_records;
    Items<Translation> translations;

    UnitIndexerData<W, M> unit_indexer_data;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !universe_type.empty()
               && universe_index.size() == universe_type.size()
               && ((!volume_ids.empty() && !logic_ints.empty() && !reals.empty())
                   || surface_types.empty())
               && !volume_records.empty() && unit_indexer_data;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OrangeParamsData& operator=(OrangeParamsData<W2, M2> const& other)
    {
        scalars = other.scalars;

        universe_type = other.universe_type;
        universe_index = other.universe_index;
        simple_unit = other.simple_unit;

        surface_ids = other.surface_ids;
        volume_ids = other.volume_ids;
        real_ids = other.real_ids;
        logic_ints = other.logic_ints;
        reals = other.reals;
        surface_types = other.surface_types;
        connectivities = other.connectivities;
        volume_records = other.volume_records;
        translations = other.translations;
        unit_indexer_data = other.unit_indexer_data;

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

    // Dimensions {num_tracks}
    StateItems<LevelId> level;
    StateItems<LevelId> next_level;

    // Dimensions {num_tracks, max_level}
    StateItems<Real3>      pos;
    StateItems<Real3>      dir;
    StateItems<VolumeId>   vol;
    StateItems<UniverseId> universe;

    // Surface crossing, dimensions {num_tracks, max_level}
    StateItems<SurfaceId>      surf;
    StateItems<Sense>          sense;
    StateItems<BoundaryResult> boundary;

    // Scratch space
    Items<Sense> temp_sense;  // [track][max_faces]
    Items<FaceId> temp_face;  // [track][max_intersections]
    Items<real_type> temp_distance;  // [track][max_intersections]
    Items<size_type> temp_isect;  // [track][max_intersections]

    //// METHODS ////

    //! True if sizes are consistent and nonzero
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return !level.empty()
            && next_level.size() == level.size()
            && !pos.empty()
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
        level      = other.level;
        next_level = other.next_level;
        pos        = other.pos;
        dir        = other.dir;
        vol        = other.vol;
        universe   = other.universe;
        surf       = other.surf;
        sense      = other.sense;
        boundary   = other.boundary;

        temp_sense = other.temp_sense;
        temp_face = other.temp_face;
        temp_distance = other.temp_distance;
        temp_isect = other.temp_isect;

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
                   HostCRef<OrangeParamsData> const& params,
                   size_type size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(num_tracks > 0);

    resize(&data->level, num_tracks);
    resize(&data->next_level, num_tracks);

    auto size = params.scalars.max_level * num_tracks;

    resize(&data->pos, size);
    resize(&data->dir, size);
    resize(&data->vol, size);

    resize(&data->universe, size);
    resize(&data->surf, size);
    resize(&data->sense, size);
    resize(&data->boundary, size);

    size_type face_states = params.scalars.max_faces * num_tracks;
    resize(&data->temp_sense, face_states);

    size_type isect_states = params.scalars.max_intersections * num_tracks;
    resize(&data->temp_face, isect_states);
    resize(&data->temp_distance, isect_states);
    resize(&data->temp_isect, isect_states);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
