//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/CelerOpticalOffload.cc
//---------------------------------------------------------------------------//
#include "CelerOpticalOffload.hh"

#include <DDG4/Factories.h>
#include <G4Cerenkov.hh>
#include <G4ProcessManager.hh>
#include <G4Scintillation.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>

#include "corecel/io/Logger.hh"
#include "corecel/math/UnitUtils.hh"
#include "geocel/g4/Convert.hh"
#include "accel/LocalOpticalGenOffload.hh"
#include "accel/detail/IntegrationSingleton.hh"

namespace celeritas
{
namespace dd
{
//---------------------------------------------------------------------------//
/*!
 * Standard constructor
 */
CelerOpticalOffload::CelerOpticalOffload(dd4hep::sim::Geant4Context* ctxt,
                                         std::string const& name)
    : dd4hep::sim::Geant4SteppingAction(ctxt, name)
{
    CELER_LOG(info) << "Registered CelerOpticalOffload stepping action";
}

//---------------------------------------------------------------------------//
/*!
 * Stepping action to offload optical distributions to Celeritas.
 */
void CelerOpticalOffload::operator()(G4Step const* step, G4SteppingManager*)
{
    CELER_EXPECT(step);

    constexpr double clhep_time{1 / units::nanosecond};
    constexpr double clhep_length{1 / units::centimeter};

    auto& local = detail::IntegrationSingleton::instance().local_offload();
    if (!local)
    {
        // Offloading is disabled
        return;
    }

    if (step->GetStepLength() == 0)
    {
        // Skip "no-process"-defined steps
        return;
    }

    auto* pm = step->GetTrack()->GetDefinition()->GetProcessManager();
    CELER_ASSERT(pm);

    // Determine how many Cherenkov and scintillation photons to generate
    size_type num_cherenkov{0};
    size_type num_scintillation{0};

    if (auto const* p = dynamic_cast<G4Cerenkov const*>(pm->GetProcess("Cerenk"
                                                                       "ov")))
    {
        num_cherenkov = p->GetNumPhotons();
    }
    if (auto const* p = dynamic_cast<G4Scintillation const*>(
            pm->GetProcess("Scintillation")))
    {
        num_scintillation = p->GetNumPhotons();
    }

    if (num_cherenkov == 0 && num_scintillation == 0)
    {
        return;
    }

    auto* pre_step = step->GetPreStepPoint();
    auto* post_step = step->GetPostStepPoint();
    CELER_ASSERT(pre_step && post_step);

    // Create distribution and push to Celeritas
    // TODO: Get optical material ID from geometry
    optical::GeneratorDistributionData data;
    data.time = convert_from_geant(post_step->GetGlobalTime(), clhep_time);
    data.step_length = convert_from_geant(step->GetStepLength(), clhep_length);
    data.charge = units::ElementaryCharge{
        static_cast<real_type>(post_step->GetCharge())};
    data.material = OptMatId(0);  // TODO: map from G4Material to OptMatId
    data.points[StepPoint::pre]
        = {units::LightSpeed(pre_step->GetBeta()),
           convert_from_geant(pre_step->GetPosition(), clhep_length)};
    data.points[StepPoint::post]
        = {units::LightSpeed(post_step->GetBeta()),
           convert_from_geant(post_step->GetPosition(), clhep_length)};

    auto& gen_offload = dynamic_cast<LocalOpticalGenOffload&>(local);

    if (num_cherenkov > 0)
    {
        data.type = GeneratorType::cherenkov;
        data.num_photons = num_cherenkov;
        CELER_ASSERT(data);
        gen_offload.Push(data);
    }
    if (num_scintillation > 0)
    {
        data.type = GeneratorType::scintillation;
        data.num_photons = num_scintillation;
        CELER_ASSERT(data);
        gen_offload.Push(data);
    }

    CELER_LOG_LOCAL(debug) << "Offloading " << num_cherenkov
                           << " Cherenkov and " << num_scintillation
                           << " scintillation photons";
}

//---------------------------------------------------------------------------//
}  // namespace dd
}  // namespace celeritas

DECLARE_GEANT4ACTION_NS(celeritas::dd, CelerOpticalOffload)
