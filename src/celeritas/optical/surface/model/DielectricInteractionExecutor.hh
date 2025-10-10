//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/DielectricInteractionExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "DielectricInteractionData.hh"
#include "SurfaceInteraction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * \todo This is a placeholder executor that does nothing until the dielectric
 * and UNIFIED calculators are implemented.
 */
struct DielectricInteractionExecutor
{
    NativeCRef<DielectricData> dielectric_data;
    NativeCRef<UnifiedReflectionData> reflection_data;

    inline CELER_FUNCTION SurfaceInteraction operator()(CoreTrackView const&) const
    {
        return SurfaceInteraction::from_absorption();
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
