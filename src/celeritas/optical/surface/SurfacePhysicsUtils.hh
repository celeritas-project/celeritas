//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Whether a track is entering the surface defined by the given normal.
 *
 * The surface normal convention used in Celeritas optical physics is that
 * the normal direction points opposite the incident track direction. This
 * function makes checks for this condition explicit in the code.
 */
inline CELER_FUNCTION bool
is_entering_surface(Real3 const& dir, Real3 const& normal)
{
    return dot_product(dir, normal) < 0;
}

//---------------------------------------------------------------------------//
/*!
 * Get the next track surface position in the given direction.
 *
 * Type-safe operation to ensure direction is only added in track-local frames.
 * Uses unsigned underflow when moving reverse (dir = -1) while on a
 * pre-surface (pos = 0) to wrap to an invalid position value.
 */
CELER_FORCEINLINE_FUNCTION SurfaceTrackPosition
advance_subsurface_position_along(SurfaceTrackPosition pos,
                                  SubsurfaceDirection dir)
{
    return pos + to_signed_offset(dir);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a valid facet normal by wrapping a roughness calculator.
 *
 * Some facet normal calculators might not produce surface normals valid for
 * optical physics surface crossings (see \c is_entering_surface ). This
 * functor will construct and repeatedly sample the distribution until the
 * track is exiting the sampled facet normal.
 */
template<class Calculator>
class EnteringSurfaceNormalSampler
{
  public:
    template<class... Args>
    CELER_FUNCTION EnteringSurfaceNormalSampler(Real3 const& dir,
                                                Real3 const& normal,
                                                Args&&... args)
        : dir_{dir}, sample_normal_{normal, celeritas::forward<Args>(args)...}
    {
        CELER_EXPECT(is_entering_surface(normal, dir));
    }

    // Repeatedly sample facet normal until satisfies entering surface
    template<class Engine>
    CELER_FUNCTION Real3 operator()(Engine& rng)
    {
        Real3 local_normal;
        do
        {
            local_normal = sample_normal_(rng);
        } while (!is_entering_surface(local_normal, dir_));
        return local_normal;
    }

  private:
    Real3 const& dir_;
    Calculator sample_normal_;
};

//---------------------------------------------------------------------------//
/*!
 * Helper functor to use an integer-typed result based on subsurface direction,
 * calculated without branching.
 *
 * Returns the supplied value when the direction is equal to the template
 * parameter, and zero otherwise.
 */
template<SubsurfaceDirection D, class IntType>
struct IfDirectionEquals
{
    IntType value;

    //! Return value if direction matches template parameter
    inline CELER_FUNCTION IntType operator()(SubsurfaceDirection dir) const
    {
        return static_cast<IntType>(D == dir) * value;
    }
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

template<class IntType>
using IfForwardDirection
    = IfDirectionEquals<SubsurfaceDirection::forward, IntType>;

template<class IntType>
using IfReverseDirection
    = IfDirectionEquals<SubsurfaceDirection::reverse, IntType>;

//---------------------------------------------------------------------------//
/*!
 * Wrapper for an \c ItemMap that can be oriented.
 *
 * If the map is forward-oriented, then it behaves the same as an \c ItemMap.
 * If the map is reverse-oriented, then it behaves the same as a reverse
 * iterator over \c ItemMap.
 */
template<class T, class U>
class OrientedItemMap
{
  public:
    //! Construct from ItemMap and orientation
    inline CELER_FUNCTION
    OrientedItemMap(ItemMap<T, U> const& map, SubsurfaceDirection orientation)
        : map_(map), orientation_(orientation)
    {
    }

    // Map surface track position, taking orientation into account
    inline CELER_FUNCTION U operator[](T pos) const
    {
        T index{IfReverseDirection<size_type>{map_.size() - 1}(orientation_)
                + to_signed_offset(orientation_) * pos.get()};
        CELER_ASSERT(index < map_.size());
        return map_[index];
    }

  private:
    ItemMap<T, U> const& map_;
    SubsurfaceDirection orientation_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
