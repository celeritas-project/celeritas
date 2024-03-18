//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A generic grid of 1D data with arbitrary interpolation.
 */
struct GenericGridData
{
    ItemRange<real_type> grid;  //!< x grid
    ItemRange<real_type> value;  //!< f(x) value
    Interp grid_interp{Interp::size_};  //!< Interpolation along x
    Interp value_interp{Interp::size_};  //!< Interpolation along f(x)

    //! Whether the record is initialized and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return grid.size() >= 2 && value.size() == grid.size()
               && grid_interp != Interp::size_ && value_interp != Interp::size_;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
