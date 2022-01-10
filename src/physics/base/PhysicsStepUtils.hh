//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsStepUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "physics/material/MaterialTrackView.hh"
#include "CutoffView.hh"
#include "ParticleTrackView.hh"
#include "PhysicsTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// INLINE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
inline CELER_FUNCTION real_type
calc_tabulated_physics_step(const MaterialTrackView& material,
                            const ParticleTrackView& particle,
                            PhysicsTrackView&        physics);

template<class Engine>
inline CELER_FUNCTION ParticleTrackView::Energy
                      calc_energy_loss(const CutoffView&        cutoffs,
                                       const MaterialTrackView& material,
                                       const ParticleTrackView& particle,
                                       const PhysicsTrackView&  physics,
                                       real_type                step_length,
                                       Engine&                  rng);

struct ProcessIdModelId
{
    ParticleProcessId ppid;
    ModelId           model;

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return ppid && model; }
};

template<class Engine>
inline CELER_FUNCTION ProcessIdModelId select_process_and_model(
    const ParticleTrackView& particle, PhysicsTrackView& physics, Engine& rng);

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "PhysicsStepUtils.i.hh"
