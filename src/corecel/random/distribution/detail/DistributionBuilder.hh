//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/detail/DistributionBuilder.hh
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
struct DistributionBuilder
{
    using Real3 = Array<real_type, 3>;

#define CELER_DISTRIB_BUILD(CLS, RECORD)                      \
    CELER_FUNCTION CLS operator()(RECORD const& record) const \
    {                                                         \
        return CLS{record};                                   \
    }
    CELER_DISTRIB_BUILD(DeltaDistribution<real_type>,
                        DeltaDistributionRecord<real_type>);
    CELER_DISTRIB_BUILD(NormalDistribution<real_type>,
                        NormalDistributionRecord);
    CELER_DISTRIB_BUILD(DeltaDistribution<Real3>,
                        DeltaDistributionRecord<Real3>);
    CELER_DISTRIB_BUILD(IsotropicDistribution<real_type>,
                        IsotropicDistributionRecord);
    CELER_DISTRIB_BUILD(UniformBoxDistribution<real_type>,
                        UniformBoxDistributionRecord);
#undef CELER_DISTRIB_TRAITS
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
