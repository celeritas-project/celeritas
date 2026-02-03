//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/InterpolatorHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/grid/NonuniformGridBuilder.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/inp/MucfPhysics.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for creating interpolators for host-only muCF input data.
 *
 * \sa MucfMaterialInserter
 */
class InterpolatorHelper
{
  public:
    // Construct with grid input data
    InterpolatorHelper(inp::Grid input);

    // Interpolate data at given point
    real_type operator()(real_type value) const;

  private:
    using Items = Collection<real_type, Ownership::value, MemSpace::host>;
    using ItemsRef
        = Collection<real_type, Ownership::const_reference, MemSpace::host>;

    Items reals_;
    NonuniformGridRecord grid_record_;
    ItemsRef reals_ref_;
    NonuniformGridCalculator interpolate_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
