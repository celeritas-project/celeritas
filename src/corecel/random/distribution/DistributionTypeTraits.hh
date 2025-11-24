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

#include "detail/DistributionFromRecordBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map 1D distribution enumeration to distribution type.
 */
template<OnedDistributionType>
struct OnedDistributionTypeTraits;

#define CELER_DISTRIB_TRAITS(ENUM_VALUE, NAME, CLS)                     \
    template<>                                                          \
    struct OnedDistributionTypeTraits<OnedDistributionType::ENUM_VALUE> \
        : public EnumToClass<OnedDistributionType,                      \
                             OnedDistributionType::ENUM_VALUE,          \
                             CLS>                                       \
    {                                                                   \
        using RecordT = NAME##DistributionRecord;                       \
    }

CELER_DISTRIB_TRAITS(delta_oned, DeltaOned, DeltaDistribution<real_type>);
CELER_DISTRIB_TRAITS(normal, Normal, NormalDistribution<real_type>);

#undef CELER_DISTRIB_TRAITS

//---------------------------------------------------------------------------//
/*!
 * Map 3D distribution enumeration to distribution type.
 */
template<ThreedDistributionType>
struct ThreedDistributionTypeTraits;

#define CELER_DISTRIB_TRAITS(ENUM_VALUE, NAME, CLS)                         \
    template<>                                                              \
    struct ThreedDistributionTypeTraits<ThreedDistributionType::ENUM_VALUE> \
        : public EnumToClass<ThreedDistributionType,                        \
                             ThreedDistributionType::ENUM_VALUE,            \
                             CLS>                                           \
    {                                                                       \
        using RecordT = NAME##DistributionRecord;                           \
    }

CELER_DISTRIB_TRAITS(delta_threed,
                     DeltaThreed,
                     DeltaDistribution<DeltaThreedDistributionRecord::Real3>);
CELER_DISTRIB_TRAITS(isotropic, Isotropic, IsotropicDistribution<real_type>);
CELER_DISTRIB_TRAITS(uniform_box, UniformBox, UniformBoxDistribution<real_type>);

#undef CELER_DISTRIB_TRAITS

//---------------------------------------------------------------------------//
/*!
 * Expand a macro to a switch statement over all possible distribution types.
 */
struct DistributionVisitor
{
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

    detail::DistributionFromRecordBuilder build_distribution;

#define CELER_DISTRIB_CASE(LOWER, UPPER)                       \
    case OnedDistributionType::LOWER:                          \
        return celeritas::forward<F>(func)(build_distribution( \
            params.LOWER##_records[ItemId<UPPER##DistributionRecord>(idx)]))
    switch (type)
    {
        CELER_DISTRIB_CASE(delta_oned, DeltaOned);
        CELER_DISTRIB_CASE(normal, Normal);
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

    detail::DistributionFromRecordBuilder build_distribution;

#define CELER_DISTRIB_CASE(LOWER, UPPER)                       \
    case ThreedDistributionType::LOWER:                        \
        return celeritas::forward<F>(func)(build_distribution( \
            params.LOWER##_records[ItemId<UPPER##DistributionRecord>(idx)]))
    switch (type)
    {
        CELER_DISTRIB_CASE(delta_threed, DeltaThreed);
        CELER_DISTRIB_CASE(isotropic, Isotropic);
        CELER_DISTRIB_CASE(uniform_box, UniformBox);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
#undef CELER_DISTRIB_CASE
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
