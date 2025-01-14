//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/TrackerVisitor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "orange/OrangeData.hh"

#include "RectArrayTracker.hh"
#include "SimpleUnitTracker.hh"
#include "UniverseTypeTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply a functor to a universe tracker of unknown type.
 *
 * An instance of this class is like \c std::visit but accepting a UniverseId
 * rather than a \c std::variant .
 *
 * Example: \code
 TrackerVisitor visit_tracker{params_};
 auto new_pos = visit_tracker(
    [&pos](auto&& u) { return u.initialize(pos); },
    univ_id);
 \endcode
 */
class TrackerVisitor
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<OrangeParamsData>;
    //!@}

  public:
    // Construct from ORANGE params
    explicit inline CELER_FUNCTION TrackerVisitor(ParamsRef const& params);

    // Apply the function to the universe specified by the given ID
    template<class F>
    CELER_FUNCTION decltype(auto) operator()(F&& func, UniverseId id);

  private:
    ParamsRef const& params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from ORANGE params.
 */
CELER_FUNCTION TrackerVisitor::TrackerVisitor(ParamsRef const& params)
    : params_(params)
{
}

#if !defined(__DOXYGEN__) || __DOXYGEN__ > 0x010908
//---------------------------------------------------------------------------//
/*!
 * Apply the function to the universe specified by the given ID.
 */
template<class F>
CELER_FUNCTION decltype(auto)
TrackerVisitor::operator()(F&& func, UniverseId id)
{
    CELER_EXPECT(id < params_.universe_types.size());
    size_type universe_idx = params_.universe_indices[id];

    // Apply type-deleted functor based on type
    return visit_universe_type(
        [&](auto u_traits) {
            using UTraits = decltype(u_traits);
            using UId = OpaqueId<typename UTraits::record_type>;
            using Tracker = typename UTraits::tracker_type;
            return func(Tracker{params_, UId{universe_idx}});
        },
        params_.universe_types[id]);
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
