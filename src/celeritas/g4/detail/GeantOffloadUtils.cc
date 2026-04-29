//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/detail/GeantOffloadUtils.cc
//---------------------------------------------------------------------------//
#include "GeantOffloadUtils.hh"

#include <G4Step.hh>

#include "geocel/GeantGeoParams.hh"
#include "geocel/GeoOpticalIdMap.hh"
#include "celeritas/ext/GeantStepView.hh"
#include "celeritas/optical/gen/GeneratorData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Populate a \c GeneratorStepData with \c G4StepPoint data.
 */
optical::GeneratorStepData data_from_point(GeantStepPointView const& p)
{
    optical::GeneratorStepData data;
    data.speed = p.speed();
    data.time = native_value_from(p.time());
    data.pos = native_value_from(p.pos());
    return data;
}

//---------------------------------------------------------------------------//
/*!
 * Populate a \c GeneratorDistributionData with \c G4Step data.
 *
 * The global Geant4 geometry should be loaded so that the optical material may
 * be determined from the Geant4 material.
 *
 * The generator type and number of photons is not populated, and should be
 * initialized by the caller.
 */
optical::GeneratorDistributionData distribution_from_step(G4Step const& g4_step)
{
    auto geant_geo = celeritas::global_geant_geo().lock();
    CELER_VALIDATE(geant_geo, << "global Geant4 geometry is not loaded");

    GeantStepView step{const_cast<G4Step&>(g4_step)};
    CELER_ASSERT(step.has_step_point(StepPoint::pre)
                 && step.has_step_point(StepPoint::post));

    auto pre_step = step.pre_step();
    auto post_step = step.post_step();

    optical::GeneratorDistributionData data;
    data.step_length = native_value_from(step.step_length());
    data.charge = post_step.charge();

    data.points[StepPoint::pre] = data_from_point(pre_step);
    data.points[StepPoint::post] = data_from_point(post_step);

    auto* g4mat = g4_step.GetPreStepPoint()->GetMaterial();
    CELER_ASSERT(g4mat);
    data.material
        = (*geant_geo->geo_optical_id_map())[geant_geo->geant_to_id(*g4mat)];

    return data;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
