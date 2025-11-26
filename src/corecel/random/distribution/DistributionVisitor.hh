//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/DistributionVisitor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/random/data/DistributionData.hh"

#include "DistributionTypeTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Expand a macro to a switch statement over all possible distribution types.
 */
class DistributionVisitor
{
  public:
    // Construct with a reference to distribution data
    explicit inline CELER_FUNCTION
    DistributionVisitor(NativeCRef<DistributionParamsData> const&);

    // Apply a functor to a 1D dietribution
    template<class F>
    CELER_CONSTEXPR_FUNCTION decltype(auto)
    operator()(F&& func, OnedDistributionId id);

    // Apply a functor to a 3D dietribution
    template<class F>
    CELER_CONSTEXPR_FUNCTION decltype(auto)
    operator()(F&& func, ThreedDistributionId id);

  private:
    //// TYPES ////

    using ODT = OnedDistributionType;
    using TDT = ThreedDistributionType;

    //// DATA ////

    NativeCRef<DistributionParamsData> const& params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to distribution data.
 */
CELER_FUNCTION DistributionVisitor::DistributionVisitor(
    NativeCRef<DistributionParamsData> const& params)
    : params_(params)
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Expand a macro to a switch statement over all 1D distribution types.
 */
template<class F>
CELER_CONSTEXPR_FUNCTION decltype(auto)
DistributionVisitor::operator()(F&& func, OnedDistributionId id)
{
    CELER_EXPECT(id < params_.oned_types.size());

    ODT type = params_.oned_types[id];
    size_type idx = params_.oned_indices[id];

#define CELER_DISTRIB_CASE(ENUM, FIELD)                                     \
    case ODT::ENUM: {                                                       \
        using TypeT = typename OnedDistributionTypeTraits<ODT::ENUM>::type; \
        return celeritas::forward<F>(func)(                                 \
            TypeT{params_.FIELD[ItemId<TypeT::RecordT>(idx)]});             \
    }
    switch (type)
    {
        CELER_DISTRIB_CASE(delta, delta_real);
        CELER_DISTRIB_CASE(normal, normal);
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
    CELER_EXPECT(id < params_.threed_types.size());

    TDT type = params_.threed_types[id];
    size_type idx = params_.threed_indices[id];

#define CELER_DISTRIB_CASE(ENUM, FIELD)                                       \
    case TDT::ENUM: {                                                         \
        using TypeT = typename ThreedDistributionTypeTraits<TDT::ENUM>::type; \
        return celeritas::forward<F>(func)(                                   \
            TypeT{params_.FIELD[ItemId<TypeT::RecordT>(idx)]});               \
    }
    switch (type)
    {
        CELER_DISTRIB_CASE(delta, delta_real3);
        CELER_DISTRIB_CASE(isotropic, isotropic);
        CELER_DISTRIB_CASE(uniform_box, uniform_box);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
#undef CELER_DISTRIB_CASE
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Helper function for sampling from a distribution.
 */
template<class Visitor, class Dist, class Engine>
CELER_FUNCTION decltype(auto)
sample_with(Visitor&& visit, Dist&& dist, Engine& rng)
{
    return celeritas::forward<Visitor>(visit)(
        [&rng](auto&& d) -> decltype(auto) { return d(rng); },
        celeritas::forward<Dist>(dist));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
