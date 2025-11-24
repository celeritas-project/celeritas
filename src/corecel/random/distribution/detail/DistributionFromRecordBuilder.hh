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

#define CELER_DISTRIB_BUILD(CLS, NAME)                                    \
    CELER_FUNCTION CLS operator()(NAME##DistributionRecord const& record) \
        const                                                             \
    {                                                                     \
        return CLS{record};                                               \
    }
    CELER_DISTRIB_BUILD(DeltaDistribution<real_type>, DeltaOned);
    CELER_DISTRIB_BUILD(NormalDistribution<real_type>, Normal);
    CELER_DISTRIB_BUILD(DeltaDistribution<Real3>, DeltaThreed);
    CELER_DISTRIB_BUILD(IsotropicDistribution<real_type>, Isotropic);
    CELER_DISTRIB_BUILD(UniformBoxDistribution<real_type>, UniformBox);
#undef CELER_DISTRIB_TRAITS
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
