//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"

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
GeantBremsstrahlungProcess::GeantBremsstrahlungProcess(
    ModelSelection selection, double seltzer_berger_limit)
    : G4VEnergyLossProcess("eBrem"), model_selection_(selection)
{
    CELER_VALIDATE(selection != ModelSelection::none,
                   << "Cannot initialize GeantBremsstrahlungProcess with "
                      "BremsModelSelection::none.");

    auto const& em_parameters = G4EmParameters::Instance();
    double energy_min = em_parameters->MinKinEnergy();
    double energy_max = em_parameters->MaxKinEnergy();
    sb_limit_ = clamp(seltzer_berger_limit, energy_min, energy_max);

    if (selection == ModelSelection::relativistic
        && seltzer_berger_limit > energy_min)
    {
        CELER_LOG(warning)
            << "Bremmstrahlung without a model at low energies "
               "is not supported: extending relativistic model down to "
            << energy_min << " MeV";
        sb_limit_ = energy_min;
    }
    else if (selection == ModelSelection::seltzer_berger)
    {
        CELER_LOG(warning)
            << "Using bremmstrahlung without a relativistic "
               "model may result in failures for high energy tracks";
    }

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
        em_model->SetHighEnergyLimit(sb_limit_);
        em_model->SetSecondaryThreshold(em_parameters->BremsstrahlungTh());
#if G4VERSION_NUMBER < 1120
        em_model->SetLPMFlag(false);
#endif
        G4VEnergyLossProcess::AddEmModel(1, em_model, fluctuation_model);
        CELER_LOG(debug) << "Added G4SeltzerBergerModel from " << energy_min
                         << " to " << sb_limit_ << " MeV";

        ++model_index;
    }

    if (model_selection_ == ModelSelection::relativistic
        || model_selection_ == ModelSelection::all)
    {
        if (energy_max > sb_limit_)
        {
            if (!G4VEnergyLossProcess::EmModel(model_index))
            {
                G4VEnergyLossProcess::SetEmModel(
                    new G4eBremsstrahlungRelModel());
            }

            auto* em_model = G4VEnergyLossProcess::EmModel(model_index);
            em_model->SetLowEnergyLimit(sb_limit_);
            em_model->SetHighEnergyLimit(energy_max);
            em_model->SetSecondaryThreshold(em_parameters->BremsstrahlungTh());
#if G4VERSION_NUMBER < 1120
            em_model->SetLPMFlag(em_parameters->LPM());
#endif
            G4VEnergyLossProcess::AddEmModel(1, em_model, fluctuation_model);
            CELER_LOG(debug) << "Added G4eBremsstrahlungRelModel from "
                             << sb_limit_ << " to " << energy_max << " MeV";
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
