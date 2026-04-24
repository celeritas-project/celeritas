//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/gen/G4OffloadUtils.cc
//---------------------------------------------------------------------------//
#include "G4OffloadUtils.hh"

#include <G4Step.hh>

#include "geocel/GeantGeoParams.hh"
#include "geocel/GeoOpticalIdMap.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/optical/gen/GeneratorData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Populate a \c GeneratorStepData with \c G4StepPoint data.
 */
optical::GeneratorStepData data_from_point(G4StepPoint const* p)
{
    CELER_EXPECT(p);

    optical::GeneratorStepData data;
    data.speed = units::LightSpeed(p->GetBeta());
    data.time = native_from_geant<units::ClhepTime>(p->GetGlobalTime());
    data.pos = native_from_geant<lengthunits::ClhepLength, real_type>(
        p->GetPosition());
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
optical::GeneratorDistributionData distribution_from_step(G4Step const& step)
{
    auto geant_geo = celeritas::global_geant_geo().lock();
    CELER_VALIDATE(geant_geo, << "global Geant4 geometry is not loaded");

    auto* pre_step = step.GetPreStepPoint();
    auto* post_step = step.GetPostStepPoint();
    CELER_ASSERT(pre_step && post_step);

    optical::GeneratorDistributionData data;
    data.step_length
        = native_from_geant<lengthunits::ClhepLength>(step.GetStepLength());
    data.charge = units::ElementaryCharge{
        static_cast<real_type>(post_step->GetCharge())};

    data.points[StepPoint::pre] = data_from_point(pre_step);
    data.points[StepPoint::post] = data_from_point(post_step);

    auto* g4mat = pre_step->GetMaterial();
    CELER_ASSERT(g4mat);
    data.material
        = (*geant_geo->geo_optical_id_map())[geant_geo->geant_to_id(*g4mat)];

    return data;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
