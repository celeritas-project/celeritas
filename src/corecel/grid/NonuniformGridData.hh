//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/NonuniformGridData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A grid of increasing, sorted 1D data.
 *
 * \c derivative stores the second derivative of the interpolating cubic
 * spline. If it is non-empty, cubic spline interpolation will be used.
 * Otherwise the interpolation will be linear-linear.
 *
 * \todo Piecewise polynomial spline interpolation is currently unsupported.
 */
struct NonuniformGridRecord
{
    ItemRange<real_type> grid;  //!< x grid
    ItemRange<real_type> value;  //!< f(x) value
    ItemRange<real_type> derivative;
    size_type spline_order{1};

    //! Whether the record is initialized and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return grid.size() >= 2 && value.size() == grid.size()
               && (derivative.empty() || grid.size() == derivative.size())
               && spline_order == 1;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
