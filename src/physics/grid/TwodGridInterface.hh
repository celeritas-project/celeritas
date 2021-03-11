//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TwodGridInterface.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Definition of a structured nonuniform 2D grid with node-centered data.
 *
 * This relies on an external Collection of reals. Data is indexed as `[x][y]`,
 * C-style row-major.
 */
struct TwodGridData
{
    ItemRange<real_type> x;      //!< x grid definition
    ItemRange<real_type> y;      //!< y grid definition
    ItemRange<real_type> values; //!< [x][y]

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return x.size() >= 2 && y.size() >= 2
               && values.size() == x.size() * y.size();
    }

    //! Get the data location for a specified x-y coordinate.
    CELER_FUNCTION ItemId<real_type> at(size_type ix, size_type iy) const
    {
        CELER_EXPECT(ix < this->x.size());
        CELER_EXPECT(iy < this->y.size());
        return ItemId<real_type>{this->values.front().get()
                                 + ix * this->y.size() + iy};
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
