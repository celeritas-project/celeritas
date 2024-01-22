//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/AlongStepKernels.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreTrackDataFwd.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Apply MSC step limiter (UrbanMsc)
void launch_limit_msc_step(ExplicitActionInterface const& action,
                           DeviceCRef<UrbanMscData> const& msc_data,
                           CoreParams const& params,
                           CoreState<MemSpace::device>& state);

//---------------------------------------------------------------------------//
//! Apply linear propagation
void launch_propagate(ExplicitActionInterface const& action,
                      CoreParams const& params,
                      CoreState<MemSpace::device>& state);

//---------------------------------------------------------------------------//
//! Apply MSC scattering (UrbanMsc)
void launch_apply_msc(ExplicitActionInterface const& action,
                      DeviceCRef<UrbanMscData> const& msc_data,
                      CoreParams const& params,
                      CoreState<MemSpace::device>& state);

//---------------------------------------------------------------------------//
//! Update track times
void launch_update_time(ExplicitActionInterface const& action,
                        CoreParams const& params,
                        CoreState<MemSpace::device>& state);

//---------------------------------------------------------------------------//
//! Apply energy loss with fluctuations
void launch_apply_eloss(ExplicitActionInterface const& action,
                        DeviceCRef<FluctuationData> const& fluct,
                        CoreParams const& params,
                        CoreState<MemSpace::device>& state);

//---------------------------------------------------------------------------//
//! Apply energy loss without fluctuations
void launch_apply_eloss(ExplicitActionInterface const& action,
                        CoreParams const& params,
                        CoreState<MemSpace::device>& state);
//---------------------------------------------------------------------------//
//! Update the track state at the end of along-step
void launch_update_track(ExplicitActionInterface const& action,
                         CoreParams const& params,
                         CoreState<MemSpace::device>& state);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
