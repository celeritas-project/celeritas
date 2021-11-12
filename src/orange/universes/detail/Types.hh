//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Types.hh"
#include "base/Span.hh"

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
template<class ValueT>
class OnTface
{
  public:
    using IdT = OpaqueId<ValueT>;

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

  private:
    IdT   id_{};
    Sense sense_{Sense::inside};
};

//! Equality of an OnFace (mostly for testing)
template<class ValueT>
CELER_CONSTEXPR_FUNCTION bool
operator==(const OnTface<ValueT>& lhs, const OnTface<ValueT>& rhs) noexcept
{
    return lhs.id() == rhs.id()
           && (!lhs || lhs.uncheckced_sense() == rhs.unchecked_sense());
}

//! Inequality for OnFace
template<class ValueT>
CELER_CONSTEXPR_FUNCTION bool
operator!=(const OnTface<ValueT>& lhs, const OnTface<ValueT>& rhs) noexcept
{
    return !(lhs == rhs);
}

using OnSurface = OnTface<struct Surface>;
using OnFace    = OnTface<struct Face>;

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
 *   X    |         | Initialized
 *   X    |   X     | Crossed surface into new volume
 *        |   X     | Initialized on a surface (reject)
 */
struct Initialization
{
    VolumeId  volume;
    OnSurface surface;

    //! Whether initialization succeeded
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(volume);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Next face ID and the distance to it.
 *
 * We may want to restructuer this if we store a vector of face/distance rather
 * than two separate vectors.
 */
struct TempNextFace
{
    FaceId*    face{nullptr};
    real_type* distance{nullptr};
    size_type  num_faces{0}; //!< "constant" in params

    explicit CELER_FORCEINLINE_FUNCTION operator bool() const
    {
        return static_cast<bool>(face);
    }
    CELER_FORCEINLINE_FUNCTION size_type size() const { return num_faces; }
};

//---------------------------------------------------------------------------//
/*!
 * Access to the local state.
 *
 * All variables (IDs, position, direction) are *local* to the given tracker.
 * Since this is passed by \em value, it is *not* expected to be modified,
 * except for the temporary storage references.
 *
 * The temporary vectors should be sufficient to store all the senses and
 * intersections in any cell.
 */
struct LocalState
{
    Real3        pos;
    Real3        dir;
    VolumeId     cell;
    OnSurface    surface;
    Span<Sense>  temp_senses;
    TempNextFace temp_next;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
