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
 * A grid of increasing, sorted 1D data with linear-linear interpolation.
 */
struct GenericGridRecord
{
    ItemRange<real_type> grid;  //!< x grid
    ItemRange<real_type> value;  //!< f(x) value

    //! Whether the record is initialized and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return grid.size() >= 2 && value.size() == grid.size();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
