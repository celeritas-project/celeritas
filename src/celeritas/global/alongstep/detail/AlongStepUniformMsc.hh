//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/AlongStepUniformMsc.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/global/alongstep/AlongStep.hh"

#include "MeanELoss.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Implementation of the "along step" action with Urban MSC and a uniform
 * magnetic field.
 */
inline CELER_FUNCTION void
along_step_uniform_msc(CoreTrackView const& track,
                       NativeCRef<UrbanMscData> const& msc,
                       UniformFieldParams const& field)
{
    return along_step(
        track,
        UrbanMsc{msc},
        [&field](ParticleTrackView const& particle, GeoTrackView* geo) {
            return make_mag_field_propagator<DormandPrinceStepper>(
                UniformField(field.field), field.options, particle, geo);
        },
        MeanELoss{});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
