//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantBremsstrahlungProcess.cc
//---------------------------------------------------------------------------//
#include "GeantBremsstrahlungProcess.hh"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <ostream>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4Electron.hh>
#include <G4EmParameters.hh>
#include <G4EmProcessSubType.hh>
#include <G4Gamma.hh>
#include <G4ParticleDefinition.hh>
#include <G4Positron.hh>
#include <G4SeltzerBergerModel.hh>
#include <G4VEmFluctuationModel.hh>
#include <G4VEmModel.hh>
#include <G4Version.hh>
#include <G4eBremsstrahlungRelModel.hh>

#include "corecel/Assert.hh"

#include "../GeantPhysicsOptions.hh"

using CLHEP::GeV;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with model selection.
 */
GeantBremsstrahlungProcess::GeantBremsstrahlungProcess(ModelSelection selection)
    : G4VEnergyLossProcess("eBrem")
    , is_initialized_(false)
    , model_selection_(selection)
{
    CELER_VALIDATE(selection != ModelSelection::none,
                   << "Cannot initialize GeantBremsstrahlungProcess with "
                      "BremsModelSelection::none.");
    SetProcessSubType(G4EmProcessSubType::fBremsstrahlung);
    SetSecondaryParticle(G4Gamma::Gamma());
    SetIonisation(false);
}

//---------------------------------------------------------------------------//
/*!
 * Define applicability based on particle definition.
 */
bool GeantBremsstrahlungProcess::IsApplicable(
    G4ParticleDefinition const& particle)
{
    return (&particle == G4Electron::Electron()
            || &particle == G4Positron::Positron());
}

//---------------------------------------------------------------------------//
/*!
 * Print documentation in html format.
 */
void GeantBremsstrahlungProcess::ProcessDescription(std::ostream& output) const
{
    output << "  Bremsstrahlung";
    G4VEnergyLossProcess::ProcessDescription(output);
}

//---------------------------------------------------------------------------//
// PROTECTED
//---------------------------------------------------------------------------//
/*!
 * Initialise process by constructing models based on \c ModelSelection .
 */
void GeantBremsstrahlungProcess::InitialiseEnergyLossProcess(
    G4ParticleDefinition const*, G4ParticleDefinition const*)
{
    if (is_initialized_)
    {
        // Nothing to do
        return;
    }

    auto const& em_parameters = G4EmParameters::Instance();

    double energy_min = em_parameters->MinKinEnergy();
    double energy_max = em_parameters->MaxKinEnergy();
    double sb_energy_limit = 1 * GeV;
    double energy_limit = std::min(energy_max, sb_energy_limit);
    G4VEmFluctuationModel* fluctuation_model = nullptr;

    std::size_t model_index = 0;

    if (model_selection_ == ModelSelection::seltzer_berger
        || model_selection_ == ModelSelection::all)
    {
        if (!G4VEnergyLossProcess::EmModel(model_index))
        {
            G4VEnergyLossProcess::SetEmModel(new G4SeltzerBergerModel());
        }

        auto* em_model = G4VEnergyLossProcess::EmModel(model_index);
        em_model->SetLowEnergyLimit(energy_min);
        em_model->SetHighEnergyLimit(energy_limit);
        em_model->SetSecondaryThreshold(em_parameters->BremsstrahlungTh());
#if G4VERSION_NUMBER < 1120
        em_model->SetLPMFlag(false);
#endif
        G4VEnergyLossProcess::AddEmModel(1, em_model, fluctuation_model);

        ++model_index;
    }

    if (model_selection_ == ModelSelection::relativistic
        || model_selection_ == ModelSelection::all)
    {
        if (energy_max > energy_limit)
        {
            if (!G4VEnergyLossProcess::EmModel(model_index))
            {
                G4VEnergyLossProcess::SetEmModel(
                    new G4eBremsstrahlungRelModel());
            }

            auto* em_model = G4VEnergyLossProcess::EmModel(model_index);
            em_model->SetLowEnergyLimit(energy_limit);
            em_model->SetHighEnergyLimit(energy_max);
            em_model->SetSecondaryThreshold(em_parameters->BremsstrahlungTh());
#if G4VERSION_NUMBER < 1120
            em_model->SetLPMFlag(em_parameters->LPM());
#endif
            G4VEnergyLossProcess::AddEmModel(1, em_model, fluctuation_model);
        }
    }

    is_initialized_ = true;
}

//---------------------------------------------------------------------------//
/*!
 * Print class parameters.
 */
void GeantBremsstrahlungProcess::StreamProcessInfo(std::ostream& output) const
{
    if (auto* model = G4VEnergyLossProcess::EmModel(0))
    {
        auto const& param = G4EmParameters::Instance();
        double const energy_threshold = param->BremsstrahlungTh();

        output << "      LPM flag: " << param->LPM() << " for E > "
               << model->HighEnergyLimit() / GeV << " GeV";

        if (energy_threshold < std::numeric_limits<double>::max())
        {
            output << ",  VertexHighEnergyTh(GeV)= " << energy_threshold / GeV;
        }
        output << std::endl;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
