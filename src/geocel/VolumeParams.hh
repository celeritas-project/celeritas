//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/cont/Span.hh"

#include "Types.hh"

namespace celeritas
{
namespace inp
{
struct Volumes;
}

//---------------------------------------------------------------------------//
/*!
 * Define and manage a hierarchy of volumes and instances thereof.
 *
 * See the introduction to \rstref{the Geometry API section,api_geometry} for
 * a detailed description of volumes in the detector geometry description. This
 * class abstracts the graph of user-defined volumes, relating \em nodes
 * (VolumeId, aka logical volume) to \em edges (VolumeInstanceId, aka physical
 * volume) and providing the means to determine the \em path
 * (isomorphic to a VolumeUniqueInstanceId, aka touchable history) of a track
 * state. The \em root of the graph is the world volume, and the \em level of a
 * volume in the path is the distance to the root: zero for the root volume,
 * one for its direct child, etc. The maximum value of the level in any path is
 * one less than \c num_volume_levels : an array of \c VolumeId with that size
 * can represent any path.
 *
 * In conjunction with \c GeantGeoParams, this class allows conversion between
 * the Celeritas geometry implementation and the Geant4 geometry navigation.
 *
 * \internal Construction requirements:
 * - At least one volume must be defined.
 * - Material IDs are allowed to be null for testing purposes.
 *
 * \todo We should be able to easily move the ID-related methods to a
 * GPU-friendly view rather than just this metadata class. It's not needed at
 * the moment though.
 */
class VolumeParams
{
  public:
    //!@{
    //! \name Type aliases
    using VolumeMap = LabelIdMultiMap<VolumeId>;
    using VolInstMap = LabelIdMultiMap<VolumeInstanceId>;
    using SpanVolInst = Span<VolumeInstanceId const>;
    //!@}

    //!@{
    //! \name VolumeVisitor template interface
    using VolumeRef = VolumeId;
    using VolumeInstanceRef = VolumeInstanceId;
    //!@}

    using vol_level_uint = VolumeLevelId::size_type;

  public:
    // Construct from input
    explicit VolumeParams(inp::Volumes const&);

    // Construct empty volume params for unit testing: no volumes
    VolumeParams();

    //! Empty if no volumes are present (e.g., ORANGE debugging)
    bool empty() const { return v_labels_.empty(); }

    //! World volume
    VolumeId world() const { return world_; }

    //! Depth of the volume DAG (a world without children is 1)
    vol_level_uint num_volume_levels() const { return num_volume_levels_; }

    //! Number of volumes
    VolumeId::size_type num_volumes() const { return v_labels_.size(); }

    //! Number of volume instances
    VolumeInstanceId::size_type num_volume_instances() const
    {
        return vi_labels_.size();
    }

    //! Get volume metadata
    VolumeMap const& volume_labels() const { return v_labels_; }

    //! Get volume instance metadata
    VolInstMap const& volume_instance_labels() const { return vi_labels_; }

    // Find all instances of a volume (incoming edges)
    inline SpanVolInst parents(VolumeId v_id) const;

    // Get the list of daughter volumes (outgoing edges)
    inline SpanVolInst children(VolumeId v_id) const;

    // Get the geometry material of a volume
    inline GeoMatId material(VolumeId v_id) const;

    // Get the volume being instantiated (outgoing node)
    inline VolumeId volume(VolumeInstanceId vi_id) const;

  private:
    VolumeMap v_labels_;
    VolInstMap vi_labels_;

    VolumeId world_;
    vol_level_uint num_volume_levels_{0};

    std::vector<std::vector<VolumeInstanceId>> parents_;
    std::vector<std::vector<VolumeInstanceId>> children_;
    std::vector<GeoMatId> materials_;
    std::vector<VolumeId> volumes_;
};

//---------------------------------------------------------------------------//
// HACKY GLOBAL VARIABLES DURING REFACTORING
//---------------------------------------------------------------------------//
// Set non-owning reference to global canonical volumes
void global_volumes(std::shared_ptr<VolumeParams const> const&);

// Global canonical volumes: may be nullptr
std::weak_ptr<VolumeParams const> const& global_volumes();

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Find all instances of a volume (incoming edges).
 */
auto VolumeParams::parents(VolumeId v_id) const -> SpanVolInst
{
    CELER_EXPECT(v_id < parents_.size());
    return make_span(parents_[v_id.unchecked_get()]);
}

//---------------------------------------------------------------------------//
/*!
 * Get the list of daughter volumes (outgoing edges).
 */
auto VolumeParams::children(VolumeId v_id) const -> SpanVolInst
{
    CELER_EXPECT(v_id < children_.size());
    return make_span(children_[v_id.unchecked_get()]);
}

//---------------------------------------------------------------------------//
/*!
 * Get the geometry material of a volume.
 */
GeoMatId VolumeParams::material(VolumeId v_id) const
{
    CELER_EXPECT(v_id < materials_.size());
    return materials_[v_id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume being instantiated (outgoing node).
 */
VolumeId VolumeParams::volume(VolumeInstanceId vi_id) const
{
    CELER_EXPECT(vi_id < volumes_.size());
    return volumes_[vi_id.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
