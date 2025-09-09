//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/detail/GridUtils.hh
//! \brief Grid construction helpers
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/grid/SplineDerivCalculator.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/inp/Grid.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<Ownership W>
using Values = Collection<real_type, W, MemSpace::host>;

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives or set the polynomial order.
 */
template<class GridRecord>
void set_spline(Values<Ownership::value>* values,
                DedupeCollectionBuilder<real_type>& reals,
                inp::Interpolation const& interpolation,
                GridRecord& data)
{
    CELER_EXPECT(values);
    CELER_EXPECT(data);

    if (interpolation.type == InterpolationType::cubic_spline)
    {
        if (data.value.size() < SplineDerivCalculator::min_grid_size())
        {
            CELER_LOG(warning)
                << interpolation.type
                << " interpolation is not supported on a grid with size "
                << data.value.size() << ": defaulting to linear";
            return;
        }
        // Calculate second derivatives for cubic spline interpolation
        CELER_VALIDATE(interpolation.bc
                           != SplineDerivCalculator::BoundaryCondition::size_,
                       << "Boundary condition must be specified for "
                          "calculating cubic spline second derivatives");

        Values<Ownership::const_reference> ref(*values);
        auto deriv = SplineDerivCalculator(interpolation.bc)(data, ref);
        data.derivative = reals.insert_back(deriv.begin(), deriv.end());
    }
    else if (interpolation.type == InterpolationType::poly_spline)
    {
        if (data.value.size() <= interpolation.order)
        {
            CELER_LOG(warning)
                << interpolation.type << " interpolation with order "
                << interpolation.order
                << " is not supported on a grid with size "
                << data.value.size() << ": defaulting to linear";
            return;
        }
        CELER_ASSERT(interpolation.order > 1);
        data.spline_order = interpolation.order;
    }
    else
    {
        /* No spline data added */
    }
    CELER_ENSURE(data);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
