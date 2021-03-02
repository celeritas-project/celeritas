//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsStepUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "physics/material/MaterialTrackView.hh"
#include "ParticleTrackView.hh"
#include "PhysicsTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// INLINE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
inline CELER_FUNCTION void
update_physics_step(const MaterialTrackView& material,
                    const ParticleTrackView& particle,
                    PhysicsTrackView&        physics);

inline CELER_FUNCTION ParticleTrackView::Energy
                      calc_energy_loss(const ParticleTrackView& particle,
                                       const PhysicsTrackView&  physics,
                                       real_type                step_length);

template<class Engine>
inline CELER_FUNCTION ModelId select_model(const ParticleTrackView& particle,
                                           const PhysicsTrackView&  physics,
                                           Engine&                  rng);

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "PhysicsStepUtils.i.hh"
