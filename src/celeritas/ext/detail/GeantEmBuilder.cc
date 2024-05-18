//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantEmBuilder.cc
//---------------------------------------------------------------------------//
#include "GeantEmBuilder.hh"

#include <G4Alpha.hh>
#include <G4AntiLambda.hh>
#include <G4AntiNeutrinoE.hh>
#include <G4AntiNeutrinoMu.hh>
#include <G4AntiNeutron.hh>
#include <G4AntiProton.hh>
#include <G4ChargedGeantino.hh>
#include <G4CoulombScattering.hh>
#include <G4Deuteron.hh>
#include <G4EmParameters.hh>
#include <G4Geantino.hh>
#include <G4GenericIon.hh>
#include <G4HadParticles.hh>
#include <G4HadronicParameters.hh>
#include <G4He3.hh>
#include <G4KaonMinus.hh>
#include <G4KaonPlus.hh>
#include <G4Lambda.hh>
#include <G4MuIonisation.hh>
#include <G4MuMultipleScattering.hh>
#include <G4MuonMinus.hh>
#include <G4MuonPlus.hh>
#include <G4NeutrinoE.hh>
#include <G4NeutrinoMu.hh>
#include <G4Neutron.hh>
#include <G4NuclearStopping.hh>
#include <G4PhysListUtil.hh>
#include <G4PhysicsListHelper.hh>
#include <G4PionMinus.hh>
#include <G4PionPlus.hh>
#include <G4PionZero.hh>
#include <G4Proton.hh>
#include <G4Triton.hh>
#include <G4WentzelVIModel.hh>
#include <G4hBremsstrahlung.hh>
#include <G4hIonisation.hh>
#include <G4hMultipleScattering.hh>
#include <G4hPairProduction.hh>
#include <G4ionIonisation.hh>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Set up minimal EM particle list.
 */
void GeantEmBuilder::construct_minimal_em_set()
{
    // Instantiate singletons for physics
    G4PhysListUtil::InitialiseParameters();

    // Pseudo-particles
    G4Geantino::GeantinoDefinition();
    G4ChargedGeantino::ChargedGeantinoDefinition();
    G4NeutrinoMu::NeutrinoMu();
    G4AntiNeutrinoMu::AntiNeutrinoMu();
    G4NeutrinoE::NeutrinoE();
    G4AntiNeutrinoE::AntiNeutrinoE();

    // Leptons
    G4MuonPlus::MuonPlus();
    G4MuonMinus::MuonMinus();

    // Mesons
    G4PionPlus::PionPlus();
    G4PionMinus::PionMinus();
    G4PionZero::PionZero();
    G4KaonPlus::KaonPlus();
    G4KaonMinus::KaonMinus();

    // Barions
    G4Proton::Proton();
    G4AntiProton::AntiProton();
    G4Neutron::Neutron();
    G4AntiNeutron::AntiNeutron();
    G4Lambda::Lambda();
    G4AntiLambda::AntiLambda();

    // Ions
    G4Deuteron::Deuteron();
    G4Triton::Triton();
    G4He3::He3();
    G4Alpha::Alpha();
    G4GenericIon::GenericIon();
}

//---------------------------------------------------------------------------//
/*!
 * Construct lepton and hadron EM standard physics.
 */
void GeantEmBuilder::construct_charged(G4hMultipleScattering* ion_msc,
                                       G4NuclearStopping* nuclear_stopping,
                                       bool is_wvi)
{
    G4PhysicsListHelper* plh = G4PhysicsListHelper::GetPhysicsListHelper();
    G4HadronicParameters* hadron_params = G4HadronicParameters::Instance();
    bool is_hep = (G4EmParameters::Instance()->MaxKinEnergy()
                   > hadron_params->EnergyThresholdForHeavyHadrons());

    // Add mu+- physics
    this->construct_muon_em_physics(G4MuonPlus::MuonPlus(), is_hep, is_wvi);
    this->construct_muon_em_physics(G4MuonMinus::MuonMinus(), is_hep, is_wvi);

    // Add pi+- physics
    this->construct_light_hadrons(
        G4PionPlus::PionPlus(), G4PionMinus::PionMinus(), is_hep, false, is_wvi);

    // Add K+- physics
    this->construct_light_hadrons(
        G4KaonPlus::KaonPlus(), G4KaonMinus::KaonMinus(), is_hep, false, is_wvi);

    // Add p and pbar physics
    this->construct_light_hadrons(
        G4Proton::Proton(), G4AntiProton::AntiProton(), is_hep, true, is_wvi);
    if (nuclear_stopping)
    {
        plh->RegisterProcess(nuclear_stopping, G4Proton::Proton());
    }

    // Add ion physics
    this->construct_ion_em_physics(ion_msc, nuclear_stopping);

    // Add hyperons and antiparticles
    if (is_hep)
    {
        this->construct_basic_em_physics(
            ion_msc, G4HadParticles::GetHeavyChargedParticles());

        // b- and c- charged particles
        if (hadron_params->EnableBCParticles())
        {
            this->construct_basic_em_physics(
                ion_msc, G4HadParticles::GetBCChargedHadrons());
        }
        // Light hyper-nuclei
        if (hadron_params->EnableHyperNuclei())
        {
            this->construct_basic_em_physics(
                ion_msc, G4HadParticles::GetChargedHyperNuclei());
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add light hadron EM standard physics.
 */
void GeantEmBuilder::construct_light_hadrons(G4ParticleDefinition* p1,
                                             G4ParticleDefinition* p2,
                                             bool is_hep,
                                             bool is_proton,
                                             bool is_wvi)
{
    G4PhysicsListHelper* plh = G4PhysicsListHelper::GetPhysicsListHelper();

    // Multiple and single Coulomb scattering
    G4hMultipleScattering* msc = new G4hMultipleScattering();
    G4CoulombScattering* coulomb = nullptr;
    if (is_wvi)
    {
        msc->SetEmModel(new G4WentzelVIModel());
        coulomb = new G4CoulombScattering();
        plh->RegisterProcess(coulomb, p1);
    }
    plh->RegisterProcess(msc, p1);

    if (is_proton)
    {
        // Different MSC and Coulomb processes for proton/antiproton
        msc = new G4hMultipleScattering();
        if (is_wvi)
        {
            msc->SetEmModel(new G4WentzelVIModel());
            coulomb = new G4CoulombScattering();
        }
    }
    plh->RegisterProcess(msc, p2);
    if (is_wvi)
    {
        plh->RegisterProcess(coulomb, p2);
    }

    // Ionization
    plh->RegisterProcess(new G4hIonisation(), p1);
    plh->RegisterProcess(new G4hIonisation(), p2);

    // Bremsstrahlung and pair production
    if (is_hep)
    {
        G4hBremsstrahlung* brems = new G4hBremsstrahlung();
        plh->RegisterProcess(brems, p1);
        plh->RegisterProcess(brems, p2);

        G4hPairProduction* pair_prod = new G4hPairProduction();
        plh->RegisterProcess(pair_prod, p1);
        plh->RegisterProcess(pair_prod, p2);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add ion EM standard physics.
 */
void GeantEmBuilder::construct_ion_em_physics(
    G4hMultipleScattering* ion_msc, G4NuclearStopping* nuclear_stopping)
{
    G4PhysicsListHelper* plh = G4PhysicsListHelper::GetPhysicsListHelper();

    G4ParticleDefinition* deuteron = G4Deuteron::Deuteron();
    plh->RegisterProcess(ion_msc, deuteron);
    plh->RegisterProcess(new G4hIonisation(), deuteron);

    G4ParticleDefinition* triton = G4Triton::Triton();
    plh->RegisterProcess(ion_msc, triton);
    plh->RegisterProcess(new G4hIonisation(), triton);

    G4ParticleDefinition* alpha = G4Alpha::Alpha();
    plh->RegisterProcess(new G4hMultipleScattering(), alpha);
    plh->RegisterProcess(new G4ionIonisation(), alpha);
    if (nuclear_stopping)
    {
        plh->RegisterProcess(nuclear_stopping, alpha);
    }

    G4ParticleDefinition* he3 = G4He3::He3();
    plh->RegisterProcess(new G4hMultipleScattering(), he3);
    plh->RegisterProcess(new G4ionIonisation(), he3);
    if (nuclear_stopping)
    {
        plh->RegisterProcess(nuclear_stopping, he3);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add basic EM standard physics.
 */
void GeantEmBuilder::construct_basic_em_physics(G4hMultipleScattering* ion_msc,
                                                std::vector<int> const& hadrons)
{
    G4PhysicsListHelper* plh = G4PhysicsListHelper::GetPhysicsListHelper();
    G4ParticleTable* table = G4ParticleTable::GetParticleTable();

    for (auto const& pdg : hadrons)
    {
        auto p = table->FindParticle(pdg);
        if (!p || p->GetPDGCharge() == 0)
        {
            continue;
        }
        plh->RegisterProcess(ion_msc, p);
        plh->RegisterProcess(new G4hIonisation(), p);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add muon EM standard physics.
 */
void GeantEmBuilder::construct_muon_em_physics(G4ParticleDefinition* p,
                                               bool is_hep,
                                               bool is_wvi)
{
    G4PhysicsListHelper* plh = G4PhysicsListHelper::GetPhysicsListHelper();

    G4MuMultipleScattering* msc = new G4MuMultipleScattering();
    if (is_wvi)
    {
        msc->SetEmModel(new G4WentzelVIModel());
        plh->RegisterProcess(new G4CoulombScattering(), p);
    }
    plh->RegisterProcess(msc, p);
    plh->RegisterProcess(new G4MuIonisation(), p);

    if (is_hep)
    {
        plh->RegisterProcess(new G4MuBremsstrahlung(), p);
        plh->RegisterProcess(new G4MuPairProduction(), p);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
