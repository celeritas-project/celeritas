//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/detail/DistributionFromRecordBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/random/data/DistributionData.hh"

#include "../DeltaDistribution.hh"
#include "../IsotropicDistribution.hh"
#include "../NormalDistribution.hh"
#include "../UniformBoxDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a distribution object from a record,
 */
struct DistributionFromRecordBuilder
{
    using Real3 = Array<real_type, 3>;

    CELER_FUNCTION DeltaDistribution<real_type>
    operator()(DeltaOnedDistributionRecord const& record) const
    {
        return DeltaDistribution<real_type>{record.value};
    }

    CELER_FUNCTION DeltaDistribution<Real3>
    operator()(DeltaThreedDistributionRecord const& record) const
    {
        return DeltaDistribution<Real3>{record.value};
    }

    CELER_FUNCTION NormalDistribution<real_type>
    operator()(NormalDistributionRecord const& record) const
    {
        return NormalDistribution<real_type>{record.mean, record.stddev};
    }

    CELER_FUNCTION IsotropicDistribution<real_type>
    operator()(IsotropicDistributionRecord const&) const
    {
        return IsotropicDistribution<real_type>{};
    }

    CELER_FUNCTION UniformBoxDistribution<real_type>
    operator()(UniformBoxDistributionRecord const& record) const
    {
        return UniformBoxDistribution<real_type>{record.lower, record.upper};
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
