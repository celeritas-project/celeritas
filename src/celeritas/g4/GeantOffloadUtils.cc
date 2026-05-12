//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/GeantOffloadUtils.cc
//---------------------------------------------------------------------------//
#include "GeantOffloadUtils.hh"

#include <G4Step.hh>
#include <G4Track.hh>

#include "geocel/GeantGeoParams.hh"
#include "geocel/GeoOpticalIdMap.hh"
#include "celeritas/ext/GeantStepView.hh"
#include "celeritas/ext/GeantTrackView.hh"
#include "celeritas/optical/gen/GeneratorData.hh"

namespace celeritas
{
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
    CELER_EXPECT(g4_step.GetTrack());

    GeantStepView step{const_cast<G4Step&>(g4_step)};

    optical::GeneratorDistributionData data;
    data.step_length = native_value_from(step.step_length());
    data.charge = GeantTrackView{*g4_step.GetTrack()}.particle().charge();

    for (auto p : range(StepPoint::size_))
    {
        CELER_ASSERT(step.has_step_point(p));
        auto point = step.step_point(p);

        auto& data_point = data.points[p];
        data_point.speed = point.speed();
        data_point.time = native_value_from(point.time());
        data_point.pos
            = static_array_cast<real_type>(native_value_from(point.pos()));
    }

    auto geant_geo = celeritas::global_geant_geo().lock();
    CELER_VALIDATE(geant_geo, << "global Geant4 geometry is not loaded");

    auto* g4mat = g4_step.GetPreStepPoint()->GetMaterial();
    CELER_ASSERT(g4mat);
    data.material
        = (*geant_geo->geo_optical_id_map())[geant_geo->geant_to_id(*g4mat)];

    return data;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
