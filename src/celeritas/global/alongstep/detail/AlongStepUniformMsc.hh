//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/AlongStepUniformMsc.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/UniformField.hh"

#include "AlongStepNeutral.hh"
#include "EnergyLossApplier.hh"
#include "UrbanMsc.hh"

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
along_step_uniform_msc(const NativeCRef<UrbanMscData>& msc,
                       const UniformFieldParams&       field,
                       NoData,
                       CoreTrackView const& track)
{
    return along_step(
        UrbanMsc{msc},
        [&field](const ParticleTrackView& particle, GeoTrackView* geo) {
            return make_mag_field_propagator<DormandPrinceStepper>(
                UniformField(field.field), field.options, particle, geo);
        },
        EnergyLossApplier{},
        track);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
