//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/DistributionTypeTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/EnumClassUtils.hh"
#include "corecel/random/data/DistributionData.hh"

#include "detail/DistributionBuilder.hh"

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

CELER_DISTRIB_TRAITS(delta,
                     DeltaDistribution<detail::DistributionBuilder::Real3>);
CELER_DISTRIB_TRAITS(isotropic, IsotropicDistribution<real_type>);
CELER_DISTRIB_TRAITS(uniform_box, UniformBoxDistribution<real_type>);

#undef CELER_DISTRIB_TRAITS

//---------------------------------------------------------------------------//
/*!
 * Expand a macro to a switch statement over all possible distribution types.
 */
struct DistributionVisitor
{
    using Real3 = Array<real_type, 3>;

    NativeCRef<DistributionParamsData> const& params;

    template<class F>
    CELER_CONSTEXPR_FUNCTION decltype(auto)
    operator()(F&& func, OnedDistributionId id);

    template<class F>
    CELER_CONSTEXPR_FUNCTION decltype(auto)
    operator()(F&& func, ThreedDistributionId id);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Expand a macro to a switch statement over all 1D distribution types.
 */
template<class F>
CELER_CONSTEXPR_FUNCTION decltype(auto)
DistributionVisitor::operator()(F&& func, OnedDistributionId id)
{
    CELER_EXPECT(id < params.oned_types.size());

    OnedDistributionType type = params.oned_types[id];
    size_type idx = params.oned_indices[id];

    detail::DistributionBuilder build;

#define CELER_DISTRIB_CASE(ENUM_VALUE, FIELD, RECORD) \
    case OnedDistributionType::ENUM_VALUE:            \
        return celeritas::forward<F>(func)(           \
            build(params.FIELD[ItemId<RECORD>(idx)]))
    switch (type)
    {
        CELER_DISTRIB_CASE(
            delta, delta_real, DeltaDistributionRecord<real_type>);
        CELER_DISTRIB_CASE(normal, normal, NormalDistributionRecord);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
#undef CELER_DISTRIB_CASE
}

//---------------------------------------------------------------------------//
/*!
 * Expand a macro to a switch statement over all 3D distribution types.
 */
template<class F>
CELER_CONSTEXPR_FUNCTION decltype(auto)
DistributionVisitor::operator()(F&& func, ThreedDistributionId id)
{
    CELER_EXPECT(id < params.threed_types.size());

    ThreedDistributionType type = params.threed_types[id];
    size_type idx = params.threed_indices[id];

    detail::DistributionBuilder build;

#define CELER_DISTRIB_CASE(ENUM_VALUE, FIELD, RECORD) \
    case ThreedDistributionType::ENUM_VALUE:          \
        return celeritas::forward<F>(func)(           \
            build(params.FIELD[ItemId<RECORD>(idx)]))
    switch (type)
    {
        CELER_DISTRIB_CASE(delta, delta_real3, DeltaDistributionRecord<Real3>);
        CELER_DISTRIB_CASE(isotropic, isotropic, IsotropicDistributionRecord);
        CELER_DISTRIB_CASE(
            uniform_box, uniform_box, UniformBoxDistributionRecord);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
#undef CELER_DISTRIB_CASE
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
