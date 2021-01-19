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
 * Construct from shared and static data.
 */
PhysicsTrackView::PhysicsTrackView(const PhysicsParamsPointers& params,
                                   const PhysicsStatePointers&  states,
                                   ParticleDefId,
                                   MaterialDefId,
                                   ThreadId tid)
    : params_(params), states_(states), tid_(tid)
{
    CELER_EXPECT(tid_);
}

//---------------------------------------------------------------------------//
/*!
 * Select a model ID for the current track.
 *
 * An "unassigned" model ID is valid, as it might represent a special case or a
 * particle that is not undergoing an interaction.
 */
void PhysicsTrackView::model_id(ModelId id)
{
    states_.state[tid_.get()].model_id = id;
}

//---------------------------------------------------------------------------//
/*!
 * Access the model ID that has been selected for the current track.
 *
 * If no model applies (e.g. if the particle has exited the geometry) the
 * result will be the \c ModelId() which evaluates to false.
 */
ModelId PhysicsTrackView::model_id() const
{
    return states_.state[tid_.get()].model_id;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
