//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/VolumeView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access data about a single implementation volume.
 *
 * A volume is a CSG tree of surfaces. The sorted, unique surface IDs
 * comprising the volume are "faces" and can be indexed as part of this volume
 * using the \c FaceId.
 *
 * Each surface defines an "inside" space and "outside" space that correspond
 * to "negative" and "positive" values of the quadric expression's evaluation.
 * Left of a plane is negative, for example, and evaluates to "false" or
 * "inside" or "negative". The CSG tree is encoded into a vector of Reverse
 * Polish Notation-type operations (push, negate, and, or) that is evaluated at
 * tracking time to determine whether a particle is inside the volume. The
 * encoded set of operations is the \c logic accessor.
 */
class VolumeView
{
  public:
    //@{
    //! Type aliases
    using ParamsRef = NativeCRef<OrangeParamsData>;
    //@}

  public:
    // Construct with reference to persistent data
    inline CELER_FUNCTION VolumeView(ParamsRef const& params,
                                     SimpleUnitRecord const& unit_record,
                                     LocalVolumeId id);

    //// ACCESSORS ////

    // Number of surfaces bounding this volume
    CELER_FORCEINLINE_FUNCTION FaceId::size_type num_faces() const;

    // Get surface ID for a single face
    inline CELER_FUNCTION LocalSurfaceId get_surface(FaceId id) const;

    // Get the face ID of a surface if present
    inline CELER_FUNCTION FaceId find_face(LocalSurfaceId id) const;

    // Get all surface IDs for the volume
    CELER_FORCEINLINE_FUNCTION LdgSpan<LocalSurfaceId const> faces() const;

    // Get logic definition
    CELER_FORCEINLINE_FUNCTION LdgSpan<logic_int const> logic() const;

    // Get the number of total intersections
    CELER_FORCEINLINE_FUNCTION logic_int max_intersections() const;

    // Whether the volume has internal surface crossings
    CELER_FORCEINLINE_FUNCTION bool internal_surfaces() const;

    // Whether the volume is an "implicit complement"
    CELER_FORCEINLINE_FUNCTION bool implicit_vol() const;

    // Whether the safety distance can be calculated with the simple algorithm
    CELER_FORCEINLINE_FUNCTION bool simple_safety() const;

    // Whether the intersection is the closest interior surface
    CELER_FORCEINLINE_FUNCTION bool simple_intersection() const;

  private:
    ParamsRef const& params_;
    VolumeRecord const& def_;

    static inline CELER_FUNCTION VolumeRecord const&
    volume_record(ParamsRef const&,
                  SimpleUnitRecord const& unit_record,
                  LocalVolumeId id);
};

//---------------------------------------------------------------------------//
/*!
 * Construct with reference to persistent data.
 */
CELER_FUNCTION
VolumeView::VolumeView(ParamsRef const& params,
                       SimpleUnitRecord const& unit_record,
                       LocalVolumeId id)
    : params_(params), def_(VolumeView::volume_record(params, unit_record, id))
{
}

//---------------------------------------------------------------------------//
/*!
 * Number of surfaces bounding this volume.
 */
CELER_FUNCTION FaceId::size_type VolumeView::num_faces() const
{
    return def_.faces.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get the surface ID for a single face.
 *
 * This is an O(1) operation.
 */
CELER_FUNCTION LocalSurfaceId VolumeView::get_surface(FaceId id) const
{
    CELER_EXPECT(id < this->num_faces());
    auto offset = def_.faces.begin()->unchecked_get();
    offset += id.unchecked_get();
    return params_.local_surface_ids[ItemId<LocalSurfaceId>(offset)];
}

//---------------------------------------------------------------------------//
/*!
 * Find the face ID of a surface.
 *
 * - A non-empty surface ID that's among the faces in this volume will return
 *   the face ID, which is just the index of the surface ID in the list of
 *   local faces.
 * - If the given surface is not present in the volume, the result will be
 *   false.
 *
 * This is an O(log(num_faces)) operation.
 */
CELER_FUNCTION FaceId VolumeView::find_face(LocalSurfaceId surface) const
{
    CELER_EXPECT(surface);
    auto surface_list = this->faces();
    auto iter = lower_bound(surface_list.begin(), surface_list.end(), surface);
    if (iter == surface_list.end() || *iter != surface)
    {
        // Not found
        return {};
    }

    FaceId result(iter - surface_list.begin());
    CELER_ENSURE(result < this->num_faces());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get all the surface IDs corresponding to the faces of this volume.
 */
CELER_FUNCTION LdgSpan<LocalSurfaceId const> VolumeView::faces() const
{
    return params_.local_surface_ids[def_.faces];
}

//---------------------------------------------------------------------------//
/*!
 * Get logic definition.
 */
CELER_FUNCTION LdgSpan<logic_int const> VolumeView::logic() const
{
    return params_.logic_ints[def_.logic];
}

//---------------------------------------------------------------------------//
/*!
 * Get the maximum number of surface intersections.
 */
CELER_FUNCTION logic_int VolumeView::max_intersections() const
{
    return def_.max_intersections;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the volume has internal surface crossings.
 */
CELER_FUNCTION bool VolumeView::internal_surfaces() const
{
    return def_.flags & VolumeRecord::internal_surfaces;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the volume is an "implicit complement".
 */
CELER_FUNCTION bool VolumeView::implicit_vol() const
{
    return def_.flags & VolumeRecord::implicit_vol;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the safety distance can be calculated with the simple algorithm.
 */
CELER_FUNCTION bool VolumeView::simple_safety() const
{
    return def_.flags & VolumeRecord::simple_safety;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the intersection is the closest interior surface.
 */
CELER_FUNCTION bool VolumeView::simple_intersection() const
{
    return !(def_.flags
             & (VolumeRecord::internal_surfaces | VolumeRecord::implicit_vol));
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume record data for the current volume.
 *
 * This is called during construction.
 */
inline CELER_FUNCTION VolumeRecord const&
VolumeView::volume_record(ParamsRef const& params,
                          SimpleUnitRecord const& unit,
                          LocalVolumeId local_vol_id)
{
    CELER_EXPECT(local_vol_id < unit.volumes.size());
    return params.volume_records[unit.volumes[local_vol_id]];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
