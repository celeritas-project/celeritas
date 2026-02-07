//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"

#include "OrangeTypes.hh"
#include "SenseUtils.hh"

#include "detail/BIHData.hh"

namespace celeritas
{
class OrangeParams;
class VolumeParams;
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//

//! Local ID of exterior volume for unit-type universes
inline constexpr LocalVolumeId orange_exterior_volume{0};

//! ID of the top (root/global/world, level=0) universe (scene)
inline constexpr UnivId orange_global_univ{0};

//! ID of the global universe
inline constexpr UnivLevelId orange_global_univ_level{0};

//! Logic notation used for boolean expressions
inline constexpr auto orange_tracking_logic{LogicNotation::infix};

//---------------------------------------------------------------------------//
/*!
 * Scalar values particular to an ORANGE geometry instance.
 *
 * Some of these are currently needed for state sizes (univ_levels, faces,
 * intersections), others are tested during construction (csg_levels), and
 * others are only for debugging on host (geo, volume params).
 */
struct OrangeParamsScalars
{
    size_type num_univ_levels{};
    size_type max_faces{};
    size_type max_intersections{};
    size_type max_csg_levels{};

    // Canonical volume depth: maximum stacked volume instances
    vol_level_uint num_vol_levels{};

    // Soft comparison and dynamic "bumping" values
    Tolerance<> tol;

    // Raw pointers to externally owned memory for debug output
    OrangeParams const* host_geo_params{nullptr};
    VolumeParams const* host_volume_params{nullptr};

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return num_univ_levels > 0 && max_faces > 0 && max_intersections > 0
               && tol;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data for a single implementation volume definition inside a unit.
 *
 * Surface IDs are local to the unit.
 *
 * \sa VolumeView
 * \todo Rename UnitVolumeRecord?
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
 */
struct SurfacesRecord
{
    using SurfaceTypeId = ItemId<SurfaceType>;
    using RealId = ItemId<real_type>;  // Pointer to a real
    using RealIdId = ItemId<RealId>;  // Pointer to pointer to real

    ItemMap<LocalSurfaceId, SurfaceTypeId> types;
    ItemMap<LocalSurfaceId, RealIdId> data_offsets;

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
 *
 * If using a HEP geometry implementation and this unit represents multiple
 * parent/child levels of the geometry hierarchy (via "inlining" during
 * construction; see \c g4org::ProtoConstructor):
 *
 * - Local parent is a map of local-to-local impl volume IDs: the \em value is
 *   an "implementation" placement of the canonical volume that is the
 *   \em parent enclosing the canonical volume placed at the local impl volume
 *   \em key
 * - Local level for each local impl volume is the difference between the
 *   corresponding canonical volume's depth and the highest canonical volume in
 *   in this unit. The (singular) local volume associated with the most-global
 *   canonical volume in this unit has value 0, a child volume "inlined" into
 *   this unit has value 1, etc.
 *
 * If this unit does not represent a Geant4/HEP volume hierarchy, those two
 * arrays will be empty.
 */
struct SimpleUnitRecord
{
    using VolumeRecordId = ItemId<VolumeRecord>;
    using ConnectivityRecordId = ItemId<ConnectivityRecord>;
    using LocalVolumeIdId = ItemId<LocalVolumeId>;
    using VolDepthUint = ItemId<vol_level_uint>;

    // Surface data
    SurfacesRecord surfaces;
    ItemMap<LocalSurfaceId, ConnectivityRecordId> connectivity;

    // Volume data [index by LocalVolumeId]
    ItemMap<LocalVolumeId, VolumeRecordId> volumes;
    // For volume instance mapping
    ItemMap<LocalVolumeId, LocalVolumeIdId> local_parent;
    ItemMap<LocalVolumeId, VolDepthUint> local_vol_level;

    // Bounding Interval Hierarchy tree parameters
    detail::BIHTreeRecord bih_tree;

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
 * \todo These should be indexed into by UnivId, not the default
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
        return !surfaces.empty() && volumes.size() == surfaces.size();
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
 * \c univ_types to switch on the type of universe; then \c universe_indices
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
    using UnivItems = Collection<T, W, M, UnivId>;

    using RealId = SurfacesRecord::RealId;

    //// DATA ////

    // Scalar attributes
    OrangeParamsScalars scalars;

    // High-level universe definitions
    UnivItems<UnivType> univ_types;
    UnivItems<size_type> univ_indices;
    Items<SimpleUnitRecord> simple_units;
    Items<RectArrayRecord> rect_arrays;
    Items<TransformRecord> transforms;

    // Mappings used to reconstruct canonical volumes and hierarchy
    ImplVolumeItems<VolumeId> volume_ids;
    ImplVolumeItems<VolumeInstanceId> volume_instance_ids;

    // BIH tree storage
    BIHTreeData<W, M> bih_tree_data;

    // Low-level storage
    Items<LocalSurfaceId> local_surface_ids;
    Items<LocalVolumeId> local_volume_ids;
    Items<RealId> real_ids;
    Items<vol_level_uint> vl_uints;
    Items<logic_int> logic_ints;
    Items<real_type> reals;
    Items<FastReal3> fast_real3s;
    Items<SurfaceType> surface_types;
    Items<ConnectivityRecord> connectivity_records;
    Items<VolumeRecord> volume_records;
    Items<Daughter> daughters;
    Items<OrientedBoundingZoneRecord> obz_records;

    UniverseIndexerData<W, M> univ_indexer_data;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !univ_types.empty()
               && univ_indices.size() == univ_types.size()
               && !volume_ids.empty()
               && volume_ids.size() == volume_instance_ids.size()
               && (bih_tree_data || !simple_units.empty())
               && ((!local_volume_ids.empty() && !logic_ints.empty()
                    && !reals.empty())
                   || surface_types.empty())
               && !volume_records.empty() && univ_indexer_data;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OrangeParamsData& operator=(OrangeParamsData<W2, M2> const& other)
    {
        scalars = other.scalars;

        univ_types = other.univ_types;
        univ_indices = other.univ_indices;
        simple_units = other.simple_units;
        rect_arrays = other.rect_arrays;
        transforms = other.transforms;

        volume_ids = other.volume_ids;
        volume_instance_ids = other.volume_instance_ids;

        bih_tree_data = other.bih_tree_data;

        local_surface_ids = other.local_surface_ids;
        local_volume_ids = other.local_volume_ids;
        real_ids = other.real_ids;
        vl_uints = other.vl_uints;
        logic_ints = other.logic_ints;
        reals = other.reals;
        surface_types = other.surface_types;
        connectivity_records = other.connectivity_records;
        volume_records = other.volume_records;
        obz_records = other.obz_records;
        daughters = other.daughters;
        univ_indexer_data = other.univ_indexer_data;

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
    using StateItems = StateCollection<T, W, M>;
    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    // State with dimensions {num_tracks}
    StateItems<UnivLevelId> univ_level;
    StateItems<UnivLevelId> surface_univ_level;
    StateItems<LocalSurfaceId> surf;
    StateItems<Sense> sense;
    StateItems<BoundaryResult> boundary;

    // "Local" state, needed for Shift {num_tracks}
    StateItems<UnivLevelId> next_univ_level;
    StateItems<real_type> next_step;
    StateItems<LocalSurfaceId> next_surf;
    StateItems<Sense> next_sense;

    // State with dimensions {num_tracks, scalars.num_univ_levels}
    Items<Real3> pos;
    Items<Real3> dir;
    Items<LocalVolumeId> vol;
    Items<UnivId> univ;

    // Scratch space with dimensions {track}{max_intersections}
    Items<FaceId> temp_face;
    Items<real_type> temp_distance;
    Items<size_type> temp_isect;

    //// METHODS ////

    //! True if sizes are consistent and nonzero
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return !univ_level.empty()
            && surface_univ_level.size() == this->size()
            && surf.size() == this->size()
            && sense.size() == this->size()
            && boundary.size() == this->size()
            && next_univ_level.size() == this->size()
            && next_step.size() == this->size()
            && next_surf.size() == this->size()
            && next_sense.size() == this->size()
            && pos.size() >= this->size()
            && dir.size() == pos.size()
            && vol.size() == pos.size()
            && univ.size() == pos.size()
            && !temp_face.empty()
            && temp_distance.size() == temp_face.size()
            && temp_isect.size() == temp_face.size();
        // clang-format on
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const
    {
        return univ_level.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OrangeStateData& operator=(OrangeStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        univ_level = other.univ_level;
        surface_univ_level = other.surface_univ_level;
        surf = other.surf;
        sense = other.sense;
        boundary = other.boundary;

        next_univ_level = other.next_univ_level;
        next_step = other.next_step;
        next_surf = other.next_surf;
        next_sense = other.next_sense;

        pos = other.pos;
        dir = other.dir;
        vol = other.vol;
        univ = other.univ;

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

    resize(&data->univ_level, num_tracks);
    resize(&data->surface_univ_level, num_tracks);
    resize(&data->surf, num_tracks);
    resize(&data->sense, num_tracks);
    resize(&data->boundary, num_tracks);

    resize(&data->next_univ_level, num_tracks);
    resize(&data->next_step, num_tracks);
    resize(&data->next_surf, num_tracks);
    resize(&data->next_sense, num_tracks);

    size_type num_track_univ = params.scalars.num_univ_levels * num_tracks;
    resize(&data->pos, num_track_univ);
    resize(&data->dir, num_track_univ);
    resize(&data->vol, num_track_univ);
    resize(&data->univ, num_track_univ);

    size_type num_track_isect = params.scalars.max_intersections * num_tracks;
    resize(&data->temp_face, num_track_isect);
    resize(&data->temp_distance, num_track_isect);
    resize(&data->temp_isect, num_track_isect);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
