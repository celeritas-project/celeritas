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
                                            Rng&                     rng,
                                            Interaction*             result);

template<class Rng>
inline CELER_FUNCTION void move_and_select_model(const CutoffView& cutoffs,
                                                 const GeoMaterialView& geo_mat,
                                                 GeoTrackView&          geo,
                                                 MaterialTrackView&     mat,
                                                 ParticleTrackView& particle,
                                                 PhysicsTrackView&  phys,
                                                 SimTrackView&      sim,
                                                 Rng&               rng,
                                                 real_type*         edep,
                                                 Interaction*       result);

inline CELER_FUNCTION void post_process(GeoTrackView&      geo,
                                        ParticleTrackView& particle,
                                        PhysicsTrackView&  phys,
                                        SimTrackView&      sim,
                                        real_type*         edep,
                                        const Interaction& result);

//---------------------------------------------------------------------------//
} // namespace demo_loop

#include "KernelUtils.i.hh"
