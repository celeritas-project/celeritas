//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/DistributionTypeTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/EnumClassUtils.hh"
#include "corecel/random/Types.hh"
#include "corecel/random/data/DistributionData.hh"

#include "DeltaDistribution.hh"
#include "IsotropicDistribution.hh"
#include "NormalDistribution.hh"
#include "UniformBoxDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map 1D distribution enumeration to distribution type.
 */
template<OnedDistributionType>
struct OnedDistributionTypeTraits;

#define CELER_DISTRIB_TRAITS(ENUM_VALUE, CLS)                           \
    template<>                                                          \
    struct OnedDistributionTypeTraits<OnedDistributionType::ENUM_VALUE> \
        : public EnumToClass<OnedDistributionType,                      \
                             OnedDistributionType::ENUM_VALUE,          \
                             CLS>                                       \
    {                                                                   \
    }

CELER_DISTRIB_TRAITS(delta, DeltaDistribution<real_type>);
CELER_DISTRIB_TRAITS(normal, NormalDistribution<real_type>);

#undef CELER_DISTRIB_TRAITS

//---------------------------------------------------------------------------//
/*!
 * Map 3D distribution enumeration to distribution type.
 */
template<ThreedDistributionType>
struct ThreedDistributionTypeTraits;

#define CELER_DISTRIB_TRAITS(ENUM_VALUE, CLS)                               \
    template<>                                                              \
    struct ThreedDistributionTypeTraits<ThreedDistributionType::ENUM_VALUE> \
        : public EnumToClass<ThreedDistributionType,                        \
                             ThreedDistributionType::ENUM_VALUE,            \
                             CLS>                                           \
    {                                                                       \
    }

CELER_DISTRIB_TRAITS(delta, DeltaDistribution<Real3>);
CELER_DISTRIB_TRAITS(isotropic, IsotropicDistribution<real_type>);
CELER_DISTRIB_TRAITS(uniform_box, UniformBoxDistribution<real_type>);

#undef CELER_DISTRIB_TRAITS

//---------------------------------------------------------------------------//
}  // namespace celeritas
