//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/cont/Span.hh"
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Signed surface.
 *
 * Particles are never allowed to be logically "on" a surface: they must be
 * logically on one side or another so that they are in a particular volume.
 *
 * This class encapsulates the storage of the sense/value: we could try using a
 * bit field to compress the ID and sense at the cost of reducing the max
 * number of surfaces/faces by a factor of two.
 */
template<class IdT>
class OnTface
{
    static_assert(is_opaque_id_v<IdT>);

  public:
    //! Not on a surface
    constexpr OnTface() = default;

    //! On a particular side of the given surface (id may be null)
    CELER_CONSTEXPR_FUNCTION OnTface(IdT id, Sense sense) noexcept
        : id_{id}, sense_{sense}
    {
    }

    //! Whether we're on a surface
    explicit CELER_CONSTEXPR_FUNCTION operator bool() const noexcept
    {
        return static_cast<bool>(id_);
    }

    //! Get the ID of the surface/face (or "null" if not on a face)
    CELER_CONSTEXPR_FUNCTION IdT id() const noexcept { return id_; }

    //! Get the sense if we're on a face
    CELER_FUNCTION Sense sense() const
    {
        CELER_EXPECT(*this);
        return sense_;
    }

    //! Get the sense (unspecified if not on a face, to allow passthrough)
    CELER_CONSTEXPR_FUNCTION Sense unchecked_sense() const noexcept
    {
        return sense_;
    }

    //! Reverse the current sense, moving from one side to the other
    CELER_FUNCTION void flip_sense()
    {
        CELER_EXPECT(*this);
        sense_ = ::celeritas::flip_sense(sense_);
    }

  private:
    IdT id_{};
    Sense sense_{Sense::inside};
};

using OnLocalSurface = OnTface<LocalSurfaceId>;
using OnFace = OnTface<FaceId>;

//---------------------------------------------------------------------------//
/*!
 * Distance and next-surface information.
 *
 * The resulting sense is *before* crossing the boundary (on the current side
 * of it).
 */
struct Intersection
{
    OnLocalSurface surface;
    real_type distance = no_intersection();

    //! Whether a next surface has been found
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(surface);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Volume ID and surface ID after initialization.
 *
 * Possible configurations for the initialization result ('X' means 'has
 * a valid ID', i.e. evaluates to true):
 *
 *  Vol   | Surface | Description
 * :----: | :-----: | :-------------------------------
 *        |         | Failed to find new volume
 *        |   X     | Initialized on a surface (reject)
 *   X    |         | Initialized
 *   X    |   X     | Crossed surface into new volume
 */
struct Initialization
{
    LocalVolumeId volume;
    OnLocalSurface surface;

    //! Whether initialization succeeded
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(volume);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Local face IDs, distances, and ordering.
 *
 * This is temporary space for calculating the distance-to-intersection within
 * a volume. The faces and distances are effectively pairs, up to index \c
 * size.
 *
 * The index vector \c isect is initialized with the sequence `[0, size)` to
 * allow indirect sorting of the intersections stored in the face/distance
 * pairs.
 */
struct TempNextFace
{
    FaceId* face{nullptr};
    real_type* distance{nullptr};
    size_type* isect{nullptr};

    size_type size{0};  //!< Maximum number of intersections

    explicit CELER_FORCEINLINE_FUNCTION operator bool() const
    {
        return static_cast<bool>(face);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Access to the local state.
 *
 * All variables (IDs, position, direction) are \em local to the given tracker.
 * Since this is passed by \em value, it is \em not expected to be modified,
 * except for the temporary storage references.
 *
 * The temporary vectors should be sufficient to store all the senses and
 * intersections in any volume.
 *
 * \todo To enable cooperative multithreading in GPU contexts, we could rename
 * this to `LocalStateBase`, define `struct DeviceLocalState : LocalStateBase`,
 * and alias `LocalState = DeviceLocalState`. The device local state might
 * include, for example, a warp-local mask of active threads passed from the
 * core track view.
 */
struct LocalState
{
    Real3 pos;
    Real3 dir;
    LocalVolumeId volume;
    OnLocalSurface surface;
    TempNextFace temp_next;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
