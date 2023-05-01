//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/AlongStepRZMapFieldMsc.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/RZMapField.hh"
#include "celeritas/field/RZMapFieldData.hh"

#include "AlongStepNeutral.hh"
#include "FluctELoss.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Implementation of the "along step" action with Urban MSC and a r-z map
 * magnetic field (RZMapField).
 */
inline CELER_FUNCTION void
along_step_mapfield_msc(NativeCRef<UrbanMscData> const& msc,
                        NativeCRef<RZMapFieldParamsData> const& field,
                        NativeCRef<FluctuationData> const& fluct,
                        CoreTrackView const& track)
{
    return along_step(
        UrbanMsc{msc},
        [&](ParticleTrackView const& particle, GeoTrackView* geo) {
            return make_mag_field_propagator<DormandPrinceStepper>(
                RZMapField(field), field.options, particle, geo);
        },
        FluctELoss{fluct},
        track);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
