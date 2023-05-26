//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/detail/SafetyCacheTrackViewBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/math/ArrayUtils.hh"

#include "../SafetyCacheData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Provide common functionality for SafetyCacheTrackView.
 */
class SafetyCacheTrackViewBase
{
  public:
    //!@{
    //! \name Type aliases
    using Initializer_t = SafetyCacheInitializer;
    using StateRef = NativeRef<SafetyCacheStateData>;
    //!@}

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        SafetyCacheTrackViewBase const& other;  //!< Existing safety cache
        bool use_safety{true};
    };

  public:
    //! Construct with state reference
    CELER_FUNCTION
    SafetyCacheTrackViewBase(StateRef const& state, TrackSlotId tid)
        : state_{state}, tid_{tid}
    {
    }

    //! Initialize with a flag for safety calculation
    CELER_FUNCTION SafetyCacheTrackViewBase&
    operator=(Initializer_t const& init)
    {
        this->reset(init.use_safety);
        return *this;
    }

    // Initialize from another safety cache
    inline CELER_FUNCTION SafetyCacheTrackViewBase&
    operator=(DetailedInitializer const&);

    //! Whether the safety is being calculated for this track type
    CELER_FORCEINLINE_FUNCTION bool use_safety() const
    {
        return state_.radius[tid_] >= 0;
    }

    //! Whether the given position is inside the safety sphere
    inline CELER_FUNCTION bool is_inside(Real3 const& pos) const;

  protected:
    //! Clear the safety and flag if not available
    CELER_FORCEINLINE_FUNCTION void reset(bool use_safety = true)
    {
        state_.radius[tid_] = (use_safety ? 0 : -1);
    }

    //! Update the safety
    CELER_FORCEINLINE_FUNCTION void reset(Real3 const& origin, real_type radius)
    {
        CELER_EXPECT(this->use_safety());
        return this->reset_impl(origin, radius);
    }

    //! Get the cached safety sphere's radius
    CELER_FORCEINLINE_FUNCTION real_type radius() const
    {
        CELER_EXPECT(this->use_safety());
        return state_.radius[tid_];
    }

    //! Get the cache safety sphere's origin
    CELER_FORCEINLINE_FUNCTION Real3 const& origin() const
    {
        return state_.origin[tid_];
    }

    // Calculate the distance to the edge of the current safety sphere
    inline CELER_FUNCTION real_type calc_safety(Real3 const& pos) const;

  private:
    StateRef const& state_;
    TrackSlotId tid_;

    //! Update the safety (always allowed when constructing)
    CELER_FORCEINLINE_FUNCTION void
    reset_impl(Real3 const& origin, real_type radius)
    {
        CELER_EXPECT(radius >= 0);
        state_.radius[tid_] = radius;
        state_.origin[tid_] = origin;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Copy safety from another view.
 */
CELER_FUNCTION auto
SafetyCacheTrackViewBase::operator=(DetailedInitializer const& init)
    -> SafetyCacheTrackViewBase&
{
    if (!init.use_safety || !init.other.use_safety())
    {
        // We don't need the safety or one isn't available
        this->reset(init.use_safety);
    }
    else if (this != &init.other)
    {
        // Copy a valid safety from a different safety cache view
        this->reset_impl(init.other.origin(), init.other.radius());
    }
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the distance from the point to the edge of the safety sphere.
 *
 * This is just the distance from the given point to the edge of the safety
 * sphere.
 *
 * \note The input position must be within the safety sphere.
 */
CELER_FUNCTION bool SafetyCacheTrackViewBase::is_inside(Real3 const& pos) const
{
    CELER_EXPECT(this->use_safety());
    real_type to_origin_sq = celeritas::distance_sq(this->origin(), pos);
    return to_origin_sq <= ipow<2>(this->radius());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the distance from the point to the edge of the safety sphere.
 *
 * This is just the distance from the given point to the edge of the safety
 * sphere.
 *
 * \note The input position must be within the safety sphere.
 */
CELER_FUNCTION real_type
SafetyCacheTrackViewBase::calc_safety(Real3 const& pos) const
{
    CELER_EXPECT(this->use_safety());
    if (this->radius() == 0)
        return 0;

    real_type to_origin = celeritas::distance(this->origin(), pos);
    CELER_EXPECT(to_origin <= this->radius());
    return this->radius() - to_origin;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
