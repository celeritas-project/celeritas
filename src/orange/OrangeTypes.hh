//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypes.hh
//! \brief Type definitions for ORANGE geometry.
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/NumericLimits.hh"
#include "geocel/Types.hh"  // IWYU pragma: export

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T>
class BoundingBox;

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Real type used for acceleration
using fast_real_type = float;

//! Integer type for volume CSG tree representation
using logic_int = size_type;

//! Integer type for canonical volume level
using vol_level_uint = VolumeLevelId::size_type;

//! Helper class for some template dispatch functions
template<Axis T>
using AxisTag = std::integral_constant<Axis, T>;

//// ID TYPES ////

//! Identifier for a BIHNode objects
using BIHNodeId = OpaqueId<struct BIHNode_>;

//! Identifier for a daughter universe
using DaughterId = OpaqueId<struct Daughter>;

//! Identifier for a face within a volume
using FaceId = OpaqueId<struct Face_, logic_int>;

//! Bounding box used for acceleration
using FastBBox = BoundingBox<fast_real_type>;

//! Identifier for a bounding box used for acceleration
using FastBBoxId = OpaqueId<FastBBox>;

//! Identifier for an array of length three of floating point values
using FastReal3 = Array<float, 3>;

//! Local identifier for a surface within a universe
using LocalSurfaceId = OpaqueId<struct LocalSurface_>;

//! Local identifier for a geometry volume within a universe
using LocalVolumeId = OpaqueId<struct LocalVolume_>;

//! Identifier for an OrientedBoundingZone
using OrientedBoundingZoneId = OpaqueId<struct OrientedBoundingZoneRecord>;

//! Opaque index for "simple unit" data
using SimpleUnitId = OpaqueId<struct SimpleUnitRecord>;

//! Opaque index for rectilinear array data
using RectArrayId = OpaqueId<struct RectArrayRecord>;

//! Identifier for a translation of a single embedded universe
using TransformId = OpaqueId<struct TransformRecord>;

//! Identifier for a relocatable set of volumes
using UnivId = OpaqueId<struct Universe_>;

//! universe level, not necessarily canonical volume level
using UnivLevelId = OpaqueId<struct UnivLevel_, vol_level_uint>;

//// DEPRECATED ALIASES (to be removed in v1.0) ////

using LevelId [[deprecated("use UnivLevelId")]] = UnivLevelId;

using UniverseId [[deprecated("use UnivId")]] = UnivId;

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
 * These are ordered roughly by complexity. The storage requirement for
 * corresponding surfaces are:
 * - 1 for `p.|sc|c.c`,
 * - 3 for `c.`,
 * - 4 for `[ps]|k.`,
 * - 7 for `sq`, and
 * - 10 for `gq`.
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
    inv,  //!< Involute
    size_  //!< Sentinel value for number of surface types
};

//---------------------------------------------------------------------------//
/*!
 * Enumeration for mapping transform implementations to integers.
 */
enum class TransformType : unsigned char
{
    no_transformation,  //!< Identity transform
    translation,  //!< Translation only
    transformation,  //!< Translation plus rotation
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Enumeration for type-deleted universe storage.
 *
 * See \c orange/univ/UnivTypeTraits.hh for how these map to data and
 * classes.
 */
enum class UnivType : unsigned char
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
 * As an implementation detail, the "on" case is currently *exact*, but future
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
 * When crossing a boundary, whether the track is entering or exiting the
 * current boundary.
 *
 * After moving to a boundary, the track is considered `entering` the boundary.
 * Changing direction while on a boundary will change whether the track is
 * `entering` or `exiting` relative to the surface normal. When
 * `cross_boundary` is called, the track is only relocated to the new volume if
 * it is `entering` the boundary, after which it is considered `exiting`.
 */
enum class BoundaryResult : bool
{
    entering,
    exiting
};

//---------------------------------------------------------------------------//
/*!
 * Chirality of a twirly object (currently only Involute).
 */
enum class Chirality : bool
{
    left,  //!< Sinistral, spiraling counterclockwise
    right,  //!< Dextral, spiraling clockwise
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
//! Special logical Evaluator tokens ordered by precedence.
// The enum values are set to the highest 6 values of logic_int.
enum OperatorToken : logic_int
{
    lbegin = logic_int(~logic_int(6)),
    lopen = lbegin,  //!< Open parenthesis
    lclose,  //!< Close parenthesis
    lor,  //!< Binary logical OR
    land,  //!< Binary logical AND
    lnot,  //!< Unary negation
    ltrue,  //!< Push 'true'
    lend
};
}  // namespace logic

//---------------------------------------------------------------------------//
/*!
 * Masking priority.
 *
 * This is currently not implemented in GPU ORANGE except for the special
 * "background" cell and "exterior".
 */
enum class ZOrder : size_type
{
    invalid = 0,  //!< Invalid region
    background,  //!< Implicit fill
    media,  //!< Material-filled region or array
    array,  //!< Lattice array of nested arrangement
    hole,  //!< Another universe masking this one
    implicit_exterior = size_type(-2),  //!< Exterior in lower universe
    exterior = size_type(-1),  //!< The global problem boundary
};

//---------------------------------------------------------------------------//
// STRUCTS
//---------------------------------------------------------------------------//
/*!
 * Data specifying a daughter universe embedded in a volume.
 */
struct Daughter
{
    UnivId univ_id;
    TransformId trans_id;
};

//---------------------------------------------------------------------------//
/*!
 * Tolerance for construction and runtime bumping.
 *
 * The relative error is used for comparisons of magnitudes of values, and the
 * absolute error provides a lower bound for the comparison tolerance. In most
 * cases (see \c SoftEqual, \c BoundingBoxBumper , \c detail::BumpCalculator)
 * the tolerance used is a maximum of the absolute error and the 1- or 2-norm
 * of some spatial coordinate. In other cases (\c SurfaceSimplifier, \c
 * SoftSurfaceEqual) the similarity between surfaces is determined by solving
 * for a change in surface coefficients that results in no more than a change
 * in \f$ \epsilon \f$ of a particle intercept. A final special case (the \c
 * sqrt_quadratic static variable) is used to approximate the degenerate
 * condition \f$ a\sim 0\f$ for a particle traveling nearly parallel to a
 * quadric surface: see \c CylAligned for a discussion.
 *
 * The absolute error should typically be constructed from the relative error
 * (since computers use floating point precision) and a characteristic length
 * scale for the problem being used. For detector/reactor problems the length
 * might be ~1 cm, for microbiology it might be ~1 um, and for astronomy might
 * be ~1e6 m.
 *
 * \note For historical reasons, the absolute tolerance used by \c SoftEqual
 * defaults to 1/100 of the relative tolerance, whereas with \c Tolerance the
 * equivalent behavior is setting a length scale of 0.01.
 *
 * \todo Move this to a separate file.
 */
template<class T = ::celeritas::real_type>
struct Tolerance
{
    using real_type = T;

    real_type rel{};  //!< Relative error for differences
    real_type abs{};  //!< Absolute error [native length]

    //! Intercept tolerance for parallel-to-quadric cases
    static CELER_CONSTEXPR_FUNCTION real_type sqrt_quadratic()
    {
        if constexpr (std::is_same_v<real_type, double>)
            return 1e-5;
        else if constexpr (std::is_same_v<real_type, float>)
            return 5e-2f;
    }

    //! True if tolerances are valid
    CELER_CONSTEXPR_FUNCTION operator bool() const
    {
        return rel > 0 && rel < 1 && abs > 0;
    }

    // Construct from the default relative tolerance (sqrt(precision))
    static Tolerance from_default(real_type length = 1);

    // Construct from the default "soft equivalence" relative tolerance
    static Tolerance from_softequal();

    // Construct from a relative tolerance and a length scale
    static Tolerance from_relative(real_type rel, real_type length = 1);
};

extern template struct Tolerance<float>;
extern template struct Tolerance<double>;

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST/DEVICE)
//---------------------------------------------------------------------------//
/*!
 * Change whether a boundary crossing is reentrant or exiting.
 */
[[nodiscard]] CELER_CONSTEXPR_FUNCTION BoundaryResult
flip_boundary(BoundaryResult orig)
{
    return static_cast<BoundaryResult>(!static_cast<bool>(orig));
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
// Get a string corresponding to a surface sense
char const* to_cstring(Sense);

// Get a string corresponding to a surface type
char const* to_cstring(SurfaceType);

// Get a string corresponding to a transform type
char const* to_cstring(TransformType);

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
    return is_operator_token(tok) ? "()|&~*"[tok - lbegin] : '\a';
}
}  // namespace logic

// Get a string corresponding to a z ordering
char const* to_cstring(ZOrder);

// Get a printable character corresponding to a z ordering
char to_char(ZOrder z);

// Convert a printable character to a z ordering
ZOrder to_zorder(char c);

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
