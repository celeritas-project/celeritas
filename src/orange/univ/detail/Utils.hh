//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"

#include "../VolumeView.hh"
#include "Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// FUNCTION-LIKE CLASSES
//---------------------------------------------------------------------------//
/*!
 * Predicate for partitioning valid (finite positive) from invalid distances.
 */
struct IsFinite
{
    CELER_FORCEINLINE_FUNCTION bool operator()(real_type distance) const
    {
        return distance < numeric_limits<real_type>::max();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Predicate for selecting distances closer to or equal to a maximum.
 */
class IsNotFurtherThan
{
  public:
    explicit CELER_FORCEINLINE_FUNCTION IsNotFurtherThan(real_type md)
        : max_dist_(md)
    {
    }

    CELER_FORCEINLINE_FUNCTION bool operator()(real_type distance) const
    {
        return distance <= max_dist_;
    }

  private:
    real_type max_dist_;
};

//---------------------------------------------------------------------------//
/*!
 * Calculate the bump distance for a position.
 */
class BumpCalculator
{
  public:
    explicit CELER_FORCEINLINE_FUNCTION
    BumpCalculator(OrangeParamsScalars const& scalars)
        : scalars_(scalars)
    {
    }

    inline CELER_FUNCTION real_type operator()(Real3 const& pos) const
    {
        real_type result = scalars_.bump_abs;
        for (real_type p : pos)
        {
            result = celeritas::max(result, scalars_.bump_rel * std::fabs(p));
        }
        CELER_ENSURE(result > 0);
        return result;
    }

  private:
    OrangeParamsScalars const& scalars_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Convert an OnLocalSurface (may be null) to an OnFace using a volume view.
 */
inline CELER_FUNCTION OnFace find_face(VolumeView const& vol,
                                       OnLocalSurface surf)
{
    return {surf ? vol.find_face(surf.id()) : FaceId{}, surf.unchecked_sense()};
}

//---------------------------------------------------------------------------//
/*!
 * Convert an OnFace (may be null) to an OnLocalSurface using a volume view.
 */
inline CELER_FUNCTION OnLocalSurface get_surface(VolumeView const& vol,
                                                 OnFace face)
{
    return {face ? vol.get_surface(face.id()) : LocalSurfaceId{},
            face.unchecked_sense()};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
