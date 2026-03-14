//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Types.hh
//! Shared (VecGeom + ORANGE) geometry type definitions.
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Two-dimensional cartesian coordinates
using Real2 = Array<real_type, 2>;

//! Two-dimensional extents
using Size2 = Array<size_type, 2>;

//! Alias for a small square dense matrix
template<class T, size_type N>
using SquareMatrix = Array<Array<T, N>, N>;

//! Alias for a small square dense matrix
using SquareMatrixReal3 = SquareMatrix<real_type, 3>;

//---------------------------------------------------------------------------//

//! Opaque index for mapping volume-specific "sensitive detector" objects
using DetectorId = OpaqueId<struct Detector_>;

//! Identifier for a material fill
using GeoMatId = OpaqueId<struct GeoMaterial_>;

//! Combined boundary/interface surface identifier
using SurfaceId = OpaqueId<struct Surface_, unsigned int>;

//! Identifier for a canonical geometry volume that may be repeated
using VolumeId = OpaqueId<struct Volume_, unsigned int>;

//! Identifier for an instance of a geometry volume (aka physical/placed)
using VolumeInstanceId = OpaqueId<struct VolumeInstance_, unsigned int>;

//! Type-safe depth in the canonical volume graph (zero for world)
using VolumeLevelId = OpaqueId<struct VolumeLevel_, unsigned int>;

//! Identifier for a unique volume in global space (aka touchable)
using VolumeUniqueInstanceId = OpaqueId<struct VolumeInstance_, ull_int>;

//---------------------------------------------------------------------------//
//!@{
//! \name Geometry-specific implementation details

//! Implementation detail surface (for surface-based geometries)
using ImplSurfaceId = OpaqueId<struct ImplSurface_>;

//! Implementation detail: "global" volume index internal to a geometry
using ImplVolumeId = OpaqueId<struct ImplVolumeId_>;

//!@}
//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
/*!
 * Enumeration for cartesian axes.
 */
enum class Axis
{
    x,  //!< X axis/I index coordinate
    y,  //!< Y axis/J index coordinate
    z,  //!< Z axis/K index coordinate
    size_  //!< Sentinel value for looping over axes
};

//---------------------------------------------------------------------------//
/*!
 * Geometry state as a track moves across boundaries through the geometry.
 *
 * The "incoming" (incident) and "outgoing" (exiting) states are relative to
 * the \em boundary (surface) that the track is on, \em not the volume.
 *
 * \note The numeric values of the enumeration are chosen to optimize the free
 * functions \c is_valid (invalid values are strictly negative) and \c
 * is_on_boundary (values on the boundary are strictly positive).
 */
enum class GeoStatus : signed char
{
    error = -2,  //!< Unrecoverable error occurred
    invalid = -1,  //!< Unusable but allowable state
    interior = 0,  //!< In a volume, not on a boundary
    boundary_inc = 1,  //!< On a boundary, pointing into surface
    boundary_out = 2,  //!< On a boundary, pointing away from surface
};

//---------------------------------------------------------------------------//
// STRUCTS
//---------------------------------------------------------------------------//
/*!
 * Data required to initialize a geometry state.
 */
struct GeoTrackInitializer
{
    Real3 pos{0, 0, 0};
    Real3 dir{0, 0, 0};
    TrackSlotId parent;

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return dir[0] != 0 || dir[1] != 0 || dir[2] != 0;
    }

    // Constructors
    inline CELER_FUNCTION GeoTrackInitializer();
    inline CELER_FUNCTION GeoTrackInitializer(Real3, Real3);
    inline CELER_FUNCTION GeoTrackInitializer(Real3, Real3, TrackSlotId);
};

//---------------------------------------------------------------------------//
/*!
 * Result of a propagation step.
 *
 * The boundary flag means that the geometry is step limiting, but the surface
 * crossing must be called externally.
 */
struct Propagation
{
    real_type distance{0};  //!< Distance traveled
    bool boundary{false};  //!< True if hit a boundary before given distance
    bool looping{false};  //!< True if track is looping in the field propagator
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
//! Default constructor
CELER_FUNCTION GeoTrackInitializer::GeoTrackInitializer() = default;

//! Construct with an invalid parent ID
CELER_FUNCTION GeoTrackInitializer::GeoTrackInitializer(Real3 p, Real3 d)
    : GeoTrackInitializer(p, d, {})
{
}

//! Construct with position, direction, and parent ID
CELER_FUNCTION
GeoTrackInitializer::GeoTrackInitializer(Real3 p, Real3 d, TrackSlotId p_id)
    : pos(p), dir(d), parent(p_id)
{
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//

// Get a string corresponding to a geometry track state
char const* to_cstring(GeoStatus value);

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
//! Convert Axis enum value to int
CELER_CONSTEXPR_FUNCTION int to_int(Axis a)
{
    return static_cast<int>(a);
}

//---------------------------------------------------------------------------//
//! Convert int to Axis enum value
inline CELER_FUNCTION Axis to_axis(int a)
{
    CELER_EXPECT(a >= 0 && a < 3);
    return static_cast<Axis>(a);
}

//---------------------------------------------------------------------------//
//! Get the lowercase name of the axis.
inline constexpr char to_char(Axis ax)
{
    return "xyz\a"[static_cast<int>(ax)];
}

//---------------------------------------------------------------------------//
//! Whether the geometry is on a boundary
CELER_CONSTEXPR_FUNCTION bool is_valid(GeoStatus s)
{
    return static_cast<char>(s) >= 0;
}

//---------------------------------------------------------------------------//
//! Whether the geometry is on a boundary
CELER_CONSTEXPR_FUNCTION bool is_on_boundary(GeoStatus s)
{
    return static_cast<char>(s) > 0;
}

//---------------------------------------------------------------------------//
//! Whether a volume is outside the canonical geometry extents
CELER_CONSTEXPR_FUNCTION bool is_outside(VolumeId v)
{
    return !v;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
