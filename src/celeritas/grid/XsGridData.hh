//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/XsGridData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/UniformGridData.hh"
#include "celeritas/Types.hh"
#include "celeritas/UnitTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Parameterization of a discrete scalar field on a given 1D grid.
 *
 * For all  \code i >= prime_index \endcode, the \code value[i] \endcode is
 * expected to be pre-scaled by a factor of \code energy[i] \endcode.
 *
 * Interpolation is linear-linear after transforming to log-E space and before
 * scaling the value by E (if the grid point is above prime_index).
 *
 * \c derivative stores the second derivative of the interpolating cubic
 * spline.  If it is non-empty, spline interpolation will be used.
 */
struct XsGridData
{
    using EnergyUnits = units::Mev;
    using XsUnits = units::Native;

    //! "Special" value indicating none of the values are scaled by 1/E
    static CELER_CONSTEXPR_FUNCTION size_type no_scaling()
    {
        return size_type(-1);
    }

    UniformGridData log_energy;
    size_type prime_index{no_scaling()};
    ItemRange<real_type> value;
    ItemRange<real_type> derivative;

    //! Whether the interface is initialized and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return log_energy && (value.size() >= 2)
               && (prime_index < log_energy.size || prime_index == no_scaling())
               && log_energy.size == value.size()
               && (derivative.empty() || derivative.size() == value.size());
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
