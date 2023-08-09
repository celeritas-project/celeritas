//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypes.hh
//! \brief Type definitions for ORANGE geometry.
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <functional>
#include <utility>

#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/NumericLimits.hh"
#include "orange/BoundingBox.hh"
#include "orange/Types.hh"

#include "Types.hh"  // IWYU pragma: export

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Real type used for acceleration
using fast_real_type = float;

//! Integer type for volume CSG tree representation
using logic_int = unsigned short int;

//! Identifier for a BIHNode objects
using BIHNodeId = OpaqueId<struct BIHNode>;

//! Identifier for a daughter universe
using DaughterId = OpaqueId<struct Daughter>;

//! Identifier for a face within a volume
using FaceId = OpaqueId<struct Face>;

//! Bounding box used for acceleration
using FastBBox = BoundingBox<fast_real_type>;

//! Identifier for a bounding box used for acceleration
using FastBBoxId = OpaqueId<FastBBox>;

//! Identifier for the current "level", i.e., depth of embedded universe
using LevelId = OpaqueId<struct Level>;

//! Local identifier for a surface within a universe
using LocalSurfaceId = OpaqueId<struct LocalSurface>;

//! Local identifier for a geometry volume within a universe
using LocalVolumeId = OpaqueId<struct LocalVolume>;

//! Opaque index for "simple unit" data
using SimpleUnitId = OpaqueId<struct SimpleUnitRecord>;

//! Opaque index for rectilinear array data
using RectArrayId = OpaqueId<struct RectArrayRecord>;

//! Identifier for a translation of a single embedded universe
using TranslationId = OpaqueId<Real3>;

//! Identifier for a relocatable set of volumes
using UniverseId = OpaqueId<struct Universe>;

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
/*!
 * Whether a position is logically "inside" or "outside" a surface.
 *
 * For a plane, "outside" (true) is the "positive" sense and equivalent to
 * \f[
   \vec x \cdot \vec n >= 0
 * \f]
 * and "inside" is to the left of the plane's normal. Likewise, for a
 * sphere, "inside" is where the dot product of the position and outward normal
 * is negative.
 */
enum class Sense : bool
{
    inside,  //!< Quadric expression is less than zero
    outside,  //!< Expression is greater than zero
};

//---------------------------------------------------------------------------//
/*!
 * Enumeration for mapping surface classes to integers.
 *
 * These are ordered by number of coefficients needed for their representation:
 * 1 for `[ps].|c.o`, 3 for `c.`, 4 for `[ps]|k.`, 7 for `sq`, and 10 for `gq`.
 *
 * See \c orange/surf/SurfaceTypeTraits.hh for how these map to classes.
 */
enum class SurfaceType : unsigned char
{
    px,  //!< Plane aligned with X axis
    py,  //!< Plane aligned with Y axis
    pz,  //!< Plane aligned with Z axis
    cxc,  //!< Cylinder centered on X axis
    cyc,  //!< Cylinder centered on Y axis
    czc,  //!< Cylinder centered on Z axis
    sc,  //!< Sphere centered at the origin
    cx,  //!< Cylinder parallel to X axis
    cy,  //!< Cylinder parallel to Y axis
    cz,  //!< Cylinder parallel to Z axis
    p,  //!< General plane
    s,  //!< Sphere
    kx,  //!< Cone parallel to X axis
    ky,  //!< Cone parallel to Y axis
    kz,  //!< Cone parallel to Z axis
    sq,  //!< Simple quadric
    gq,  //!< General quadric
    size_  //!< Sentinel value for number of surface types
};

//---------------------------------------------------------------------------//
/*!
 * Enumeration for mapping transform implementations to integers.
 */
enum class TransformType : unsigned char
{
    translation,  //!< Translation only
    transformation,  //!< Translation plus rotation
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Enumeration for type-deleted universe storage.
 *
 * See \c orange/univ/UniverseTypeTraits.hh for how these map to data and
 * classes.
 */
enum class UniverseType : unsigned char
{
    simple,
    rect_array,
#if 0
    hex_array,
    dode_array,
    ...
#endif
    size_  //!< Sentinel value for number of universe types
};

//---------------------------------------------------------------------------//
/*!
 * Evaluated quadric expression allowing for distinct 'on surface' state.
 *
 * For a plane, "outside" is equivalent to
 * \f[
   \vec x \cdot \vec n > 0
 * \f]
 * and "inside" is to the left of the plane's normal (a negative dot product).
 * The exact equality to zero is literally an "edge case" but it can happen
 * with inter-universe coincident surfaces as well as carefully placed
 * particle sources and ray tracing.
 *
 * As an implementataion detail, the "on" case is currently *exact*, but future
 * changes might increase the width of "on" to a finite but small range
 * ("fuzziness").
 */
enum class SignedSense
{
    inside = -1,
    on = 0,
    outside = 1
};

//---------------------------------------------------------------------------//
/*!
 * When evaluating an intersection, whether the point is on the surface.
 *
 * This helps eliminate roundoff errors and other arithmetic issues.
 */
enum class SurfaceState : bool
{
    off = false,
    on = true
};

//---------------------------------------------------------------------------//
/*!
 * When crossing a boundary, whether the track exits the current volume.
 *
 * This is necessary due to changes in direction on the boundary due to
 * magnetic field and/or multiple scattering. We could extend this later to a
 * flag set of "volume changed" (internal non-reflective crossing), "direction
 * changed" (reflecting/periodic), "position changed" (bump/periodic).
 */
enum class BoundaryResult : bool
{
    reentrant = false,
    exiting = true
};

//---------------------------------------------------------------------------//
/*!
 * Volume logic encoding.
 *
 * This uses an *unscoped* enum inside a *namespace* so that its values can be
 * freely intermingled with other integers that represent face IDs.
 */
namespace logic
{
//! Special logical Evaluator tokens.
// The enum values are set to the highest 4 values of logic_int.
enum OperatorToken : logic_int
{
    lbegin = logic_int(~logic_int(4)),
    ltrue = lbegin,  //!< Push 'true'
    lor,  //!< Binary logical OR
    land,  //!< Binary logical AND
    lnot,  //!< Unary negation
    lend
};
}  // namespace logic

//---------------------------------------------------------------------------//
// STRUCTS
//---------------------------------------------------------------------------//
/*!
 * Data specifying a daughter universe embedded in a volume.
 */
struct Daughter
{
    UniverseId universe_id;
    TranslationId translation_id;
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST/DEVICE)
//---------------------------------------------------------------------------//
/*!
 * Convert a boolean value to a Sense enum.
 */
CELER_CONSTEXPR_FUNCTION Sense to_sense(bool s)
{
    return static_cast<Sense>(s);
}

//---------------------------------------------------------------------------//
/*!
 * Change the sense across a surface.
 */
CELER_CONSTEXPR_FUNCTION Sense flip_sense(Sense orig)
{
    return static_cast<Sense>(!static_cast<bool>(orig));
}

//---------------------------------------------------------------------------//
/*!
 * Change whether a boundary crossing is reentrant or exiting.
 */
CELER_CONSTEXPR_FUNCTION BoundaryResult flip_boundary(BoundaryResult orig)
{
    return static_cast<BoundaryResult>(!static_cast<bool>(orig));
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the sense based on the LHS expression of the quadric equation.
 *
 * This is an optimized jump-free version of:
 * \code
    return quadric == 0 ? SignedSense::on
        : quadric < 0 ? SignedSense::inside
        : SignedSense::outside;
 * \endcode
 * as
 * \code
    int gz = !(quadric <= 0) ? 1 : 0;
    int lz = quadric < 0 ? 1 : 0;
    return static_cast<SignedSense>(gz - lz);
 * \endcode
 * and compressed into a single line.
 *
 * NaN values are treated as "outside".
 */
CELER_CONSTEXPR_FUNCTION SignedSense real_to_sense(real_type quadric)
{
    return static_cast<SignedSense>(!(quadric <= 0) - (quadric < 0));
}

//---------------------------------------------------------------------------//
/*!
 * Convert a signed sense to a Sense enum.
 */
CELER_CONSTEXPR_FUNCTION Sense to_sense(SignedSense s)
{
    return Sense(static_cast<int>(s) >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Convert a signed sense to a surface state.
 */
CELER_CONSTEXPR_FUNCTION SurfaceState to_surface_state(SignedSense s)
{
    return s == SignedSense::on ? SurfaceState::on : SurfaceState::off;
}

//---------------------------------------------------------------------------//
/*!
 * Sentinel value indicating "no intersection".
 *
 * \todo There is probably a better place to put this since it's not a "type".
 * \todo A value of zero might also work since zero-length steps are
 * prohibited. But we'll need custom `min` and `min_element` in that case.
 */
CELER_CONSTEXPR_FUNCTION real_type no_intersection()
{
    return numeric_limits<real_type>::infinity();
}

//---------------------------------------------------------------------------//
/*!
 * Return the UniverseId of the highest-level (i.e., root) universe.
 */
CELER_CONSTEXPR_FUNCTION UniverseId top_universe_id()
{
    return UniverseId{0};
}

//---------------------------------------------------------------------------//
namespace logic
{
//! Whether an integer is a special logic token.
CELER_CONSTEXPR_FUNCTION bool is_operator_token(logic_int lv)
{
    return (lv >= lbegin);
}
}  // namespace logic

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//
//! Get a printable character corresponding to a sense.
inline constexpr char to_char(Sense s)
{
    return s == Sense::inside ? '-' : '+';
}

// Get a string corresponding to a surface type
char const* to_cstring(SurfaceType);

// Get a string corresponding to a surface state
inline char const* to_cstring(SurfaceState s)
{
    return s == SurfaceState::off ? "off" : "on";
}

//! Get a printable character corresponding to an operator.
namespace logic
{
inline constexpr char to_char(OperatorToken tok)
{
    return is_operator_token(tok) ? "*|&~"[tok - lbegin] : '\a';
}
}  // namespace logic

//---------------------------------------------------------------------------//
}  // namespace celeritas

//---------------------------------------------------------------------------//
// STD::HASH SPECIALIZATION FOR HOST CODE
//---------------------------------------------------------------------------//
//! \cond
namespace std
{
//! Specialization for std::hash for unordered storage.
template<>
struct hash<celeritas::Sense>
{
    using argument_type = celeritas::Sense;
    using result_type = std::size_t;
    result_type operator()(argument_type const& sense) const noexcept
    {
        return std::hash<bool>()(static_cast<bool>(sense));
    }
};
}  // namespace std
//! \endcond
