//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/UniformGridData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

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
    from_bounds(value_type front, value_type back, size_type size);
};

//---------------------------------------------------------------------------//
/*!
 * Construct from min/max and number of grid points.
 */
UniformGridData
UniformGridData::from_bounds(value_type front, value_type back, size_type size)
{
    CELER_EXPECT(size >= 2);
    CELER_EXPECT(front < back);
    UniformGridData result;
    result.size = size;
    result.front = front;
    result.back = back;
    result.delta = (back - front) / (size - 1);
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
