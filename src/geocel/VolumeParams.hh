//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeParams.hh
//---------------------------------------------------------------------------//
#pragma once

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
 * class abstracts the graph of volumes, relating \em nodes (VolumeId, aka
 * logical volume) to \em edges (VolumeInstanceId, aka physical volume) and
 * providing the means to determine the \em path (VolumeUniqueInstanceId, aka
 * touchable history) of a track state. In conjunction with \c GeantGeoParams
 * this allows conversion between the Celeritas geometry implementation and the
 * Geant4 geometry navigation.
 *
 * Input material IDs are allowed to be null for testing purposes.
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
    //!@}

  public:
    // Construct from input
    explicit VolumeParams(inp::Volumes const&);

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
    inline Span<VolumeInstanceId const> parents(VolumeId vid) const;

    // Get the list of daughter volumes (outgoing edges)
    inline Span<VolumeInstanceId const> children(VolumeId vid) const;

    // Get the geometry material of a volume
    inline GeoMatId material(VolumeId vid) const;

    // Get the volume being instantiated (outgoing node)
    inline VolumeId volume(VolumeInstanceId vid) const;

  private:
    VolumeMap v_labels_;
    VolInstMap vi_labels_;

    std::vector<std::vector<VolumeInstanceId>> parents_;
    std::vector<std::vector<VolumeInstanceId>> children_;
    std::vector<GeoMatId> materials_;
    std::vector<VolumeId> volumes_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Find all instances of a volume (incoming edges).
 */
Span<VolumeInstanceId const> VolumeParams::parents(VolumeId vid) const
{
    CELER_EXPECT(vid < parents_.size());
    return make_span(parents_[vid.unchecked_get()]);
}

//---------------------------------------------------------------------------//
/*!
 * Get the list of daughter volumes (outgoing edges).
 */
Span<VolumeInstanceId const> VolumeParams::children(VolumeId vid) const
{
    CELER_EXPECT(vid < children_.size());
    return make_span(children_[vid.unchecked_get()]);
}

//---------------------------------------------------------------------------//
/*!
 * Get the geometry material of a volume.
 */
GeoMatId VolumeParams::material(VolumeId vid) const
{
    CELER_EXPECT(vid < materials_.size());
    return materials_[vid.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume being instantiated (outgoing node).
 */
VolumeId VolumeParams::volume(VolumeInstanceId vid) const
{
    CELER_EXPECT(vid < volumes_.size());
    return volumes_[vid.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
