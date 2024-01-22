//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Types.hh
//! Shared (VecGeom + ORANGE) geometry type definitions.
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/ArrayUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Fixed-size array for 3D space
using Real3 = Array<real_type, 3>;

//! Alias for a small square dense matrix
template<class T, size_type N>
using SquareMatrix = Array<Array<T, N>, N>;

//! Alias for a small square dense matrix
using SquareMatrixReal3 = SquareMatrix<real_type, 3>;

//---------------------------------------------------------------------------//

//! Identifier for a surface (for surface-based geometries)
using SurfaceId = OpaqueId<struct Surface_>;

//! Identifier for a geometry volume
using VolumeId = OpaqueId<struct Volume_>;

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
// STRUCTS
//---------------------------------------------------------------------------//
/*!
 * Data required to initialize a geometry state.
 */
struct GeoTrackInitializer
{
    Real3 pos;
    Real3 dir;

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return is_soft_unit_vector(dir);
    }
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
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//
//! Convert Axis enum value to int
CELER_CONSTEXPR_FUNCTION int to_int(Axis a)
{
    return static_cast<int>(a);
}

//! Convert int to Axis enum value
inline CELER_FUNCTION Axis to_axis(int a)
{
    CELER_EXPECT(a >= 0 && a < 3);
    return static_cast<Axis>(a);
}

//! Get the lowercase name of the axis.
inline constexpr char to_char(Axis ax)
{
    return "xyz\a"[static_cast<int>(ax)];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
