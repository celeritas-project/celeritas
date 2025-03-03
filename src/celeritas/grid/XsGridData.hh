//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/XsGridData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/grid/UniformGridData.hh"
#include "celeritas/Types.hh"
#include "celeritas/UnitTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Tabulated cross section as a function of energy on a 1D grid.
 *
 * The upper grid values are expected to be pre-scaled by a factor of 1 / E.
 *
 * Interpolation is linear-linear or spline after transforming from log-E space
 * and before scaling the value by E (if the grid point is in the upper grid).
 */
struct XsGridRecord
{
    using EnergyUnits = units::Mev;
    using XsUnits = units::Native;

    UniformGridRecord lower;
    UniformGridRecord upper;  //!< Values scaled by 1/E

    //! Whether the record is initialized and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return (lower || upper)
               && (!lower || !upper || lower.grid.back == upper.grid.front);
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
