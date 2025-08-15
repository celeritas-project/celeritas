//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/BoundingBox.hh"

#include "OrangeTypes.hh"
#include "SenseUtils.hh"

#include "detail/BIHData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//

//! Local ID of exterior volume for unit-type universes
inline constexpr LocalVolumeId orange_exterior_volume{0};

//! ID of the top-level (global/world, level=0) universe (scene)
inline constexpr UniverseId orange_global_universe{0};

//---------------------------------------------------------------------------//
/*!
 * Scalar values particular to an ORANGE geometry instance.
 */
struct OrangeParamsScalars
{
    // Maximum universe depth, i.e., depth of the universe tree DAG: its value
    // is 1 for a non-nested geometry. It may not correspond to the depth of a
    // Geant4 geometry since we may "inline" certain logical volumes.
    size_type max_depth{};
    size_type max_faces{};
    size_type max_intersections{};
    size_type max_logic_depth{};

    // Soft comparison and dynamic "bumping" values
    Tolerance<> tol;

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return max_depth > 0 && max_faces > 0 && max_intersections > 0 && tol;
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
    ItemRange<LocalSurfaceId> faces;
    ItemRange<logic_int> logic;

    logic_int max_intersections{0};
    logic_int flags{0};
    DaughterId daughter_id;
    OrientedBoundingZoneId obz_id;

    //! \todo For KENO geometry we will need zorder

    //! Flag values (bit field)
    enum Flags : logic_int
    {
        internal_surfaces = 0x1,  //!< "Complex" distance-to-boundary
        implicit_vol = 0x2,  //!< Background/exterior volume
        simple_safety = 0x4,  //!< Fast safety calculation
        embedded_universe = 0x8  //!< Volume contains embedded universe
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
 *
 * \todo Change "types" and "data offsets" to be `ItemMap` taking local
 * surface
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
struct ConnectivityRecord
{
    ItemRange<LocalVolumeId> neighbors;
};

//---------------------------------------------------------------------------//
/*!
 * Data for a single OrientedBoundingZone.
 */
struct OrientedBoundingZoneRecord
{
    using Real3 = Array<fast_real_type, 3>;

    //! Half-widths of the inner and outer boxes
    Array<Real3, 2> half_widths;

    //! Offset from to center of inner and outer boxes to center of OBZ
    //! coordinate system
    Array<TransformId, 2> offset_ids;

    // Transformation from the OBZ coordinate system to the unit coordinate
    // system
    TransformId trans_id;

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return offset_ids[0] && offset_ids[1] && trans_id;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Class for storing offset data for RaggedRightIndexer.
 */
template<size_type N>
struct RaggedRightIndexerData
{
    using Sizes = Array<size_type, N>;
    using Offsets = Array<size_type, N + 1>;

    Offsets offsets;

    //! Construct with the an array denoting the size of each dimension.
    static RaggedRightIndexerData from_sizes(Sizes sizes)
    {
        CELER_EXPECT(sizes.size() > 0);

        Offsets offs;
        offs[0] = 0;
        for (auto i : range(N))
        {
            CELER_EXPECT(sizes[i] > 0);
            offs[i + 1] = sizes[i] + offs[i];
        }
        return RaggedRightIndexerData{offs};
    }
};

//---------------------------------------------------------------------------//
/*!
 * Type-deleted transform.
 */
struct TransformRecord
{
    using RealId = OpaqueId<real_type>;
    TransformType type{TransformType::size_};
    RealId data_offset;

    //! True if values are set
    explicit CELER_FUNCTION operator bool() const
    {
        return type != TransformType::size_ && data_offset;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Scalar data for a single "unit" of volumes defined by surfaces.
 */
struct SimpleUnitRecord
{
    using VolumeRecordId = OpaqueId<VolumeRecord>;

    // Surface data
    SurfacesRecord surfaces;
    ItemRange<ConnectivityRecord> connectivity;  // Index by LocalSurfaceId

    // Volume data [index by LocalVolumeId]
    ItemMap<LocalVolumeId, VolumeRecordId> volumes;

    // Bounding Interval Hierarchy tree parameters
    detail::BIHTree bih_tree;

    LocalVolumeId background{};  //!< Default if not in any other volume
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
 * Data for a single rectilinear array universe.
 */
struct RectArrayRecord
{
    using Dims = Array<size_type, 3>;
    using Grid = Array<ItemRange<real_type>, 3>;
    using SurfaceIndexerData = RaggedRightIndexerData<3>;

    // Daughter data [index by LocalVolumeId]
    ItemMap<LocalVolumeId, DaughterId> daughters;

    // Array data
    Dims dims;
    Grid grid;
    SurfaceIndexerData surface_indexer_data;

    //! Cursory check for validity
    explicit CELER_FUNCTION operator bool() const
    {
        return !daughters.empty() && !grid[to_int(Axis::x)].empty()
               && !grid[to_int(Axis::y)].empty()
               && !grid[to_int(Axis::z)].empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Surface and volume offsets to convert between local and global indices.
 *
 * Each collection should be of length num_universes + 1. The first entry is
 * zero and the last item should be the total number of surfaces or volumes.
 *
 * \todo These should be indexed into by UniverseId, not the default
 * OpaqueId<size_type>.
 * \todo move to detail/UniverseIndexerData
 */
template<Ownership W, MemSpace M>
struct UniverseIndexerData
{
    Collection<size_type, W, M> surfaces;
    Collection<size_type, W, M> volumes;

    template<Ownership W2, MemSpace M2>
    UniverseIndexerData& operator=(UniverseIndexerData<W2, M2> const& other)
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
 * Persistent data used by all BIH trees.
 *
 * \todo move to detail/BihTreeData
 */
template<Ownership W, MemSpace M>
struct BIHTreeData
{
    template<class T>
    using Items = Collection<T, W, M>;

    // Low-level storage
    Items<FastBBox> bboxes;
    Items<LocalVolumeId> local_volume_ids;
    Items<detail::BIHInnerNode> inner_nodes;
    Items<detail::BIHLeafNode> leaf_nodes;

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        // Note that inner_nodes may be empty for single-node trees
        return !bboxes.empty() && !local_volume_ids.empty()
               && !leaf_nodes.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    BIHTreeData& operator=(BIHTreeData<W2, M2> const& other)
    {
        bboxes = other.bboxes;
        local_volume_ids = other.local_volume_ids;
        inner_nodes = other.inner_nodes;
        leaf_nodes = other.leaf_nodes;

        CELER_ENSURE(static_cast<bool>(*this) == static_cast<bool>(other));
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent data used by ORANGE implementation.
 *
 * Most data will be accessed through the individual units, which reference
 * data in the "storage" below. The type and index for a universe ID will
 * determine the class type and data of the Tracker to instantiate. If *only*
 * simple units are present, then the \c simple_units data structure will just
 * be equal to a range (with the total number of universes present). Use
 * `universe_types` to switch on the type of universe; then `universe_indices`
 * to index into `simple_units` or `rect_arrays` or ...
 */
template<Ownership W, MemSpace M>
struct OrangeParamsData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ImplVolumeItems = Collection<T, W, M, ImplVolumeId>;
    template<class T>
    using UnivItems = Collection<T, W, M, UniverseId>;

    using RealId = SurfacesRecord::RealId;

    //// DATA ////

    // Scalar attributes
    OrangeParamsScalars scalars;

    // High-level universe definitions
    UnivItems<UniverseType> universe_types;
    UnivItems<size_type> universe_indices;
    Items<SimpleUnitRecord> simple_units;
    Items<RectArrayRecord> rect_arrays;
    Items<TransformRecord> transforms;

    // Optional map of ORANGE internal volume ID -> Celeritas volume ID
    ImplVolumeItems<VolumeId> volume_ids;
    ImplVolumeItems<VolumeInstanceId> volume_instance_ids;
    // TODO: for reconstructing hierarchy:
    // ImplVolumeItems<ImplVolumeId> parent_impl_volumes;

    // BIH tree storage
    BIHTreeData<W, M> bih_tree_data;

    // Low-level storage
    Items<LocalSurfaceId> local_surface_ids;
    Items<LocalVolumeId> local_volume_ids;
    Items<RealId> real_ids;
    Items<logic_int> logic_ints;
    Items<real_type> reals;
    Items<FastReal3> fast_real3s;
    Items<SurfaceType> surface_types;
    Items<ConnectivityRecord> connectivity_records;
    Items<VolumeRecord> volume_records;
    Items<Daughter> daughters;
    Items<OrientedBoundingZoneRecord> obz_records;

    UniverseIndexerData<W, M> universe_indexer_data;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !universe_types.empty()
               && universe_indices.size() == universe_types.size()
               && !volume_ids.empty()
               && volume_ids.size() == volume_instance_ids.size()
               && (bih_tree_data || !simple_units.empty())
               && ((!local_volume_ids.empty() && !logic_ints.empty()
                    && !reals.empty())
                   || surface_types.empty())
               && !volume_records.empty() && universe_indexer_data;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OrangeParamsData& operator=(OrangeParamsData<W2, M2> const& other)
    {
        scalars = other.scalars;

        universe_types = other.universe_types;
        universe_indices = other.universe_indices;
        simple_units = other.simple_units;
        rect_arrays = other.rect_arrays;
        transforms = other.transforms;

        volume_ids = other.volume_ids;
        volume_instance_ids = other.volume_instance_ids;

        bih_tree_data = other.bih_tree_data;

        local_surface_ids = other.local_surface_ids;
        local_volume_ids = other.local_volume_ids;
        real_ids = other.real_ids;
        logic_ints = other.logic_ints;
        reals = other.reals;
        surface_types = other.surface_types;
        connectivity_records = other.connectivity_records;
        volume_records = other.volume_records;
        obz_records = other.obz_records;
        daughters = other.daughters;
        universe_indexer_data = other.universe_indexer_data;

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

    // Note: this is duplicated from the associated OrangeParamsData .
    // It defines the stride into the preceding pseudo-2D Collections (pos,
    // dir, ..., etc.)
    size_type max_depth{0};

    // State with dimensions {num_tracks}
    StateItems<LevelId> level;
    StateItems<LevelId> surface_level;
    StateItems<LocalSurfaceId> surf;
    StateItems<Sense> sense;
    StateItems<BoundaryResult> boundary;

    // "Local" state, needed for Shift {num_tracks}
    StateItems<LevelId> next_level;
    StateItems<real_type> next_step;
    StateItems<LocalSurfaceId> next_surf;
    StateItems<Sense> next_sense;

    // State with dimensions {num_tracks, max_depth}
    Items<Real3> pos;
    Items<Real3> dir;
    Items<LocalVolumeId> vol;
    Items<UniverseId> universe;

    // Scratch space with dimensions {track}{max_faces}
    Items<SenseValue> temp_sense;

    // Scratch space with dimensions {track}{max_intersections}
    Items<FaceId> temp_face;
    Items<real_type> temp_distance;
    Items<size_type> temp_isect;

    //// METHODS ////

    //! True if sizes are consistent and nonzero
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return max_depth > 0
            && !level.empty()
            && surface_level.size() == this->size()
            && surf.size() == this->size()
            && sense.size() == this->size()
            && boundary.size() == this->size()
            && next_level.size() == this->size()
            && next_step.size() == this->size()
            && next_surf.size() == this->size()
            && next_sense.size() == this->size()
            && pos.size() == max_depth * this->size()
            && dir.size() == max_depth  * this->size()
            && vol.size() == max_depth  * this->size()
            && universe.size() == max_depth  * this->size()
            && !temp_sense.empty()
            && !temp_face.empty()
            && temp_distance.size() == temp_face.size()
            && temp_isect.size() == temp_face.size();
        // clang-format on
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const { return level.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OrangeStateData& operator=(OrangeStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        max_depth = other.max_depth;

        level = other.level;
        surface_level = other.surface_level;
        surf = other.surf;
        sense = other.sense;
        boundary = other.boundary;

        next_level = other.next_level;
        next_step = other.next_step;
        next_surf = other.next_surf;
        next_sense = other.next_sense;

        pos = other.pos;
        dir = other.dir;
        vol = other.vol;
        universe = other.universe;

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
                   size_type num_tracks)
{
    CELER_EXPECT(data);
    CELER_EXPECT(num_tracks > 0);

    data->max_depth = params.scalars.max_depth;

    resize(&data->level, num_tracks);
    resize(&data->surface_level, num_tracks);
    resize(&data->surf, num_tracks);
    resize(&data->sense, num_tracks);
    resize(&data->boundary, num_tracks);

    resize(&data->next_level, num_tracks);
    resize(&data->next_step, num_tracks);
    resize(&data->next_surf, num_tracks);
    resize(&data->next_sense, num_tracks);

    size_type level_states = params.scalars.max_depth * num_tracks;
    resize(&data->pos, level_states);
    resize(&data->dir, level_states);
    resize(&data->vol, level_states);
    resize(&data->universe, level_states);

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
