//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file XsGridInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "UniformGrid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Parameterization of a discrete scalar field on a given 1D grid.
 *
 * For all  \code i >= prime_index \endcode, the \code value[i] \endcode is
 * expected to be pre-scaled by a factor of \code energy[i] \endcode.
 *
 * \todo Later we will support multiple parameterizations of the x grid, and
 * possibly different interpolations on x and y. Currently interpolation is
 * linear-linear after transforming to log-E space and before scaling the value
 * by E (if the grid point is above prime_index).
 */
struct XsGridPointers
{
    using EnergyUnits = units::Mev;
    using XsUnits     = units::NativeUnit; // 1/cm

    UniformGridPointers   log_energy;
    size_type             prime_index{size_type(-1)};
    Span<const real_type> value;

    //! Whether the interface is initialized and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return log_energy && (value.size() >= 2)
               && (prime_index < log_energy.size
                   || prime_index == size_type(-1))
               && log_energy.size == value.size();
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
