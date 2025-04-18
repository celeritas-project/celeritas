//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/UniformGridData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/Collection.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data input for a uniform increasing grid.
 *
 * The four parameters are overconstrained -- we could omit back by calculating
 * from the front, delta, and size. In practice, though, that can introduce an
 * inconsistency into the "find" function.
 */
struct UniformGridData
{
    using value_type = ::celeritas::real_type;

    size_type size{};  //!< Number of grid edges/points
    value_type front{};  //!< Value of first grid point
    value_type back{};  //!< Value of last grid point
    value_type delta{};  //!< Grid cell width

    //! True if assigned and valid
    CELER_FUNCTION operator bool() const
    {
        return size >= 2 && delta > 0 && front < back;
    }

    //// HELPER FUNCTIONS ////

    // Construct on host from front/back
    inline static UniformGridData
    from_bounds(EnumArray<Bound, double> bounds, size_type size);
};

//---------------------------------------------------------------------------//
/*!
 * Parameterization of a discrete scalar field on a given 1D grid.
 *
 * \c derivative stores the second derivative of the interpolating cubic
 * spline. If it is non-empty, cubic spline interpolation will be used.
 *
 * \c spline_order stores the order of the piecewise polynomials used for
 * spline interpolation without continuous derivatives. The order must be
 * smaller than the grid size for effective spline interpolation. If the order
 * is set to 1, linear or cubic spline interpolation will be used.
 */
struct UniformGridRecord
{
    UniformGridData grid;
    ItemRange<real_type> value;
    ItemRange<real_type> derivative;
    size_type spline_order{1};

    //! Whether the record is initialized and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return grid && grid.size == value.size()
               && (derivative.empty() || grid.size == derivative.size())
               && spline_order > 0 && spline_order < value.size()
               && (derivative.empty() || spline_order == 1);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Construct from min/max and number of grid points.
 */
UniformGridData
UniformGridData::from_bounds(EnumArray<Bound, double> bounds, size_type size)
{
    CELER_EXPECT(size >= 2);
    CELER_EXPECT(bounds[Bound::lo] < bounds[Bound::hi]);
    UniformGridData result;
    result.size = size;
    result.front = bounds[Bound::lo];
    result.back = bounds[Bound::hi];
    result.delta = (bounds[Bound::hi] - bounds[Bound::lo]) / (size - 1);
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
