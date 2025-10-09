//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/TrivialInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreTrackView.hh"

#include "TrivialInteractionData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
struct TrivialInteractionExecutor
{
    NativeCRef<TrivialInteractionData> data;

    CELER_FUNCTION SurfaceInteraction operator()(CoreTrackView& track) const
    {
        auto sub_model_id = s_phys.interface(SurfacePhysicsOrder::interaction)
                                .internal_surface_id();

        switch (data.modes[sub_model_id])
        {
            case TrivialInteractionMode::absorb:
                return SurfaceInteraction::from_absorption();
            case TrivialInteractionMode::transmit:
                return {SurfaceInteraction::Action::refracted,
                        track.geometry().dir(),
                        track.particle().polarization()};
            case TrivialInteractionMode::backscatter:
                return {SurfaceInteraction::Action::reflected,
                        -track.geometry().dir(),
                        -track.particle().polarization()};
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
