//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Grid.hh
//---------------------------------------------------------------------------//
#pragma once

#include <limits>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/grid/GridTypes.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Interpolation options for the physics grids.
 *
 * \c order is only used for \c poly_spline interpolation and \c bc is only
 * used for \c cubic_spline interpolation.
 */
struct Interpolation
{
    using BC = SplineBoundaryCondition;

    InterpolationType type{InterpolationType::linear};
    //! Polynomial order for spline interpolation
    size_type order{1};
    //! Boundary conditions for calculating cubic spline second derivatives
    BC bc{BC::geant};
};

//---------------------------------------------------------------------------//
/*!
 * A grid of increasing, sorted 1D data.
 *
 * This is used to store tabulated physics data such as cross sections or
 * energy loss.
 */
struct Grid
{
    using VecDbl = std::vector<double>;

    VecDbl x;
    VecDbl y;
    Interpolation interpolation{};

    explicit operator bool() const
    {
        return !y.empty() && x.size() == y.size();
    }

    // Construct a constant grid with a single value
    static Grid from_constant(double y)
    {
        Grid result;
        result.x = {0, std::numeric_limits<double>::infinity()};
        result.y = {y, y};
        return result;
    }
};

//---------------------------------------------------------------------------//
/*!
 * A uniform grid of increasing, sorted 1D data.
 */
struct UniformGrid
{
    using GridBound = EnumArray<Bound, double>;
    using VecDbl = std::vector<double>;

    GridBound x{};
    VecDbl y;
    Interpolation interpolation{};

    explicit operator bool() const
    {
        return !y.empty() && x[Bound::hi] > x[Bound::lo];
    }
};

//---------------------------------------------------------------------------//
/*!
 * Uniform grids for storing macroscopic cross sections.
 *
 * The upper grid values are expected to be pre-scaled by a factor of 1 / E.
 */
struct XsGrid
{
    UniformGrid lower;
    UniformGrid upper;

    explicit operator bool() const
    {
        return (lower || upper)
               && (!lower || !upper
                   || lower.x[Bound::hi] == upper.x[Bound::lo]);
    }
};

//---------------------------------------------------------------------------//
/*!
 * An increasing, sorted 2D grid with node-centered data.
 *
 * Data is interpolated linearly and indexed as '[x][y]'.
 */
struct TwodGrid
{
    using VecDbl = std::vector<double>;

    VecDbl x;
    VecDbl y;
    VecDbl value;

    explicit operator bool() const
    {
        return !value.empty() && value.size() == x.size() * y.size();
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Equality operator, mainly for debugging.
 */
inline bool operator==(TwodGrid const& a, TwodGrid const& b)
{
    return a.x == b.x && a.y == b.y && a.value == b.value;
}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
