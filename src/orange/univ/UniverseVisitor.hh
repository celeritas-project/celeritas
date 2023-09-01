//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/UniverseVisitor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "orange/OrangeData.hh"

#include "UniverseTypeTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply a functor to a simple or rect-array univrse
 *
 * An instance of this class is like \c std::visit but accepting a UniverseId
 * rather than a \c std::variant .
 *
 * Example: \code
 UniverseVisitor visit_universe{params_};
 auto new_pos = visit_universe(
    [&pos](auto&& T) { return u.initialize(pos); },
    uid);
 \endcode
 */
class UniverseVisitor
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<OrangeParamsData>;
    //!@}

  public:
    // Construct from ORANGE params
    explicit inline CELER_FUNCTION UniverseVisitor(ParamsRef const& params);

    // Apply the function to the universe specified by the given id
    template<class F>
    CELER_FUNCTION decltype(auto) operator()(F&& func, UniverseId id);

  private:
    ParamsRef params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from ORANGE data.
 */
CELER_FUNCTION UniverseVisitor::UniverseVisitor(ParamsRef const& params)
    : params_(params)
{
}

//---------------------------------------------------------------------------//
/*!
 * Apply the function to the transform specified by the given ID.
 */
template<class F>
CELER_FUNCTION decltype(auto)
UniverseVisitor::operator()(F&& func, UniverseId id)
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

//---------------------------------------------------------------------------//
}  // namespace celeritas
