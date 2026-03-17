//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/InterpolatorHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/inp/Grid.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Host-only interpolator wrapper class for muCF input data.
 *
 * \sa MucfMaterialInserter
 */
class InterpolatorHelper
{
  public:
    // Construct with grid input data
    explicit InterpolatorHelper(inp::Grid const& input);

    // Interpolate data at given point
    real_type operator()(real_type value) const;

  private:
    using Items = Collection<real_type, Ownership::value, MemSpace::host>;

    Items reals_;
    NonuniformGridRecord grid_record_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
