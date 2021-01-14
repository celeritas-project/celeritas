//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsTrackView.i.hh
//---------------------------------------------------------------------------//
#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with from shared and staic data.
 */
PhysicsTrackView::PhysicsTrackView(const PhysicsParamsPointers& params,
                                   const PhysicsStatePointers&  states,
                                   ParticleDefId,
                                   MaterialDefId,
                                   ThreadId tid)
    : params_(params), states_(states), tid_(tid)
{
    REQUIRE(tid_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with from shared and staic data.
 */
ModelId PhysicsTrackView::model_id() const
{
    return states_.state[tid_.get()].model_id;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
