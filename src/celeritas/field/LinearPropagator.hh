//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/LinearPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "geocel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Propagate (move) a particle in a straight line.
 */
template<class GTV>
class LinearPropagator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = Propagation;
    //!@}

  public:
    //! Construct from a geo track view
    CELER_FUNCTION LinearPropagator(GTV&& track)
        : geo_(::celeritas::forward<GTV>(track))
    {
    }

    // Move track up to a user-provided distance, up to the next boundary
    inline CELER_FUNCTION result_type operator()(real_type dist);

  private:
    GTV geo_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class GTV>
CELER_FUNCTION LinearPropagator(GTV&&) -> LinearPropagator<GTV>;

//---------------------------------------------------------------------------//
/*!
 * Move track by a user-provided distance up to the next boundary.
 */
template<class GTV>
CELER_FUNCTION auto LinearPropagator<GTV>::operator()(real_type dist)
    -> result_type
{
    CELER_EXPECT(dist > 0);

    result_type result = geo_.find_next_step(dist);

    if (result.boundary)
    {
        geo_.move_to_boundary();
    }
    else
    {
        CELER_ASSERT(dist == result.distance);
        geo_.move_internal(dist);
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
