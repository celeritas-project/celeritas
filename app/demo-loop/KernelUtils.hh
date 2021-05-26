//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsStepUtils.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
// INLINE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
template<class Rng>
inline CELER_FUNCTION void calc_step_limits(const MaterialTrackView& mat,
                                            const ParticleTrackView& particle,
                                            PhysicsTrackView&        phys,
                                            SimTrackView&            sim,
                                            Rng&                     rng);

inline CELER_FUNCTION real_type propagate(GeoTrackView&           geo,
                                          const PhysicsTrackView& phys);

template<class Rng>
inline CELER_FUNCTION void
select_discrete_model(ParticleTrackView&        particle,
                      PhysicsTrackView&         phys,
                      Rng&                      rng,
                      real_type                 step,
                      ParticleTrackView::Energy eloss);

//---------------------------------------------------------------------------//
} // namespace demo_loop

#include "KernelUtils.i.hh"
