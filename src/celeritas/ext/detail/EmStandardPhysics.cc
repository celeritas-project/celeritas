//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/EmStandardPhysics.cc
//---------------------------------------------------------------------------//
#include "EmStandardPhysics.hh"

#include <G4EmParameters.hh>
#include <G4NuclearStopping.hh>
#include <G4ParticleDefinition.hh>
#include <G4PhysicsListHelper.hh>
#include <G4Version.hh>
#include <G4hMultipleScattering.hh>
#include <G4ionIonisation.hh>
#if G4VERSION_NUMBER >= 1070
#    include <G4EmBuilder.hh>
#else
#    include <G4Alpha.hh>
#    include <G4AntiLambda.hh>
#    include <G4AntiNeutrinoE.hh>
#    include <G4AntiNeutrinoMu.hh>
#    include <G4AntiNeutron.hh>
#    include <G4AntiProton.hh>
#    include <G4ChargedGeantino.hh>
#    include <G4CoulombScattering.hh>
#    include <G4Deuteron.hh>
#    include <G4EmParticleList.hh>
#    include <G4Geantino.hh>
#    include <G4GenericIon.hh>
#    include <G4He3.hh>
#    include <G4KaonMinus.hh>
#    include <G4KaonPlus.hh>
#    include <G4Lambda.hh>
#    include <G4MuIonisation.hh>
#    include <G4MuMultipleScattering.hh>
#    include <G4MuonMinus.hh>
#    include <G4MuonPlus.hh>
#    include <G4NeutrinoE.hh>
#    include <G4NeutrinoMu.hh>
#    include <G4Neutron.hh>
#    include <G4PionMinus.hh>
#    include <G4PionPlus.hh>
#    include <G4PionZero.hh>
#    include <G4Proton.hh>
#    include <G4Triton.hh>
#    include <G4WentzelVIModel.hh>
#    include <G4hBremsstrahlung.hh>
#    include <G4hIonisation.hh>
#    include <G4hPairProduction.hh>
#endif

#include "G4GenericIon.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Build list of particles.
 */
void EmStandardPhysics::ConstructParticle()
{
    SupportedEmStandardPhysics::ConstructParticle();
    this->construct_particle();
}

//---------------------------------------------------------------------------//
/*!
 * Build processes and models.
 */
void EmStandardPhysics::ConstructProcess()
{
    SupportedEmStandardPhysics::ConstructProcess();
    this->construct_process();
}

//---------------------------------------------------------------------------//
/*!
 * Set up minimal EM particle list.
 *
 * This is required to support Geant4 versions less than 10.7.0 which do not
 * have the G4EmBuilder.
 */
void EmStandardPhysics::construct_particle()
{
#if G4VERSION_NUMBER >= 1070
    G4EmBuilder::ConstructMinimalEmSet();
#else
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
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Build processes and models.
 *
 * This is required to support Geant4 versions less than 10.7.0 which do not
 * have the G4EmBuilder. This constructs the muon and hadron physics as in the
 * \c G4EmStandardPhysics::ConstructProcess() method of Geant4 version 10.6.0.
 */
void EmStandardPhysics::construct_process()
{
    auto& ph = *G4PhysicsListHelper::GetPhysicsListHelper();
#if G4VERSION_NUMBER >= 1070
    G4ParticleDefinition* particle = G4GenericIon::GenericIon();

    auto niel_energy_limit = G4EmParameters::Instance()->MaxNIELEnergy();
    G4NuclearStopping* nuclear_stopping = nullptr;
    if (niel_energy_limit > 0)
    {
        // Nuclear stopping is enabled if the energy limit is above zero
        nuclear_stopping = new G4NuclearStopping();
        nuclear_stopping->SetMaxKinEnergy(niel_energy_limit);
    }
    G4hMultipleScattering* ion_msc = new G4hMultipleScattering("ionmsc");
    G4ionIonisation* ion_ionization = new G4ionIonisation();

    ph.RegisterProcess(ion_msc, particle);
    ph.RegisterProcess(ion_ionization, particle);
    if (nuclear_stopping)
    {
        ph.RegisterProcess(nuclear_stopping, particle);
    }

    G4EmBuilder::ConstructCharged(ion_msc, nuclear_stopping);
#else
    // Muon and hadron bremsstrahlung and pair production
    G4MuBremsstrahlung* mu_brems = new G4MuBremsstrahlung();
    G4MuPairProduction* mu_pair = new G4MuPairProduction();
    G4hBremsstrahlung* pi_brems = new G4hBremsstrahlung();
    G4hPairProduction* pi_pair = new G4hPairProduction();
    G4hBremsstrahlung* ka_brems = new G4hBremsstrahlung();
    G4hPairProduction* ka_pair = new G4hPairProduction();
    G4hBremsstrahlung* prot_brems = new G4hBremsstrahlung();
    G4hPairProduction* prot_pair = new G4hPairProduction();

    // Muon & hadron multiple scattering
    G4MuMultipleScattering* mu_msc = new G4MuMultipleScattering();
    mu_msc->SetEmModel(new G4WentzelVIModel());
    G4CoulombScattering* mu_coulomb = new G4CoulombScattering();

    G4hMultipleScattering* pi_msc = new G4hMultipleScattering();
    pi_msc->SetEmModel(new G4WentzelVIModel());
    G4CoulombScattering* pi_coulomb = new G4CoulombScattering();

    G4hMultipleScattering* ka_msc = new G4hMultipleScattering();
    ka_msc->SetEmModel(new G4WentzelVIModel());
    G4CoulombScattering* ka_coulomb = new G4CoulombScattering();

    G4hMultipleScattering* ion_msc = new G4hMultipleScattering("ionmsc");

    // Add standard muon and hadron EM Processes
    G4EmParticleList particle_list;
    G4ParticleTable* table = G4ParticleTable::GetParticleTable();
    for (auto const& name : particle_list.PartNames())
    {
        G4ParticleDefinition* particle = table->FindParticle(name);
        if (!particle)
        {
            continue;
        }
        else if (name == "mu+" || name == "mu-")
        {
            ph.RegisterProcess(mu_msc, particle);
            ph.RegisterProcess(new G4MuIonisation(), particle);
            ph.RegisterProcess(mu_brems, particle);
            ph.RegisterProcess(mu_pair, particle);
            ph.RegisterProcess(mu_coulomb, particle);
        }
        else if (name == "alpha" || name == "He3")
        {
            ph.RegisterProcess(new G4hMultipleScattering(), particle);
            ph.RegisterProcess(new G4ionIonisation(), particle);
        }
        else if (name == "GenericIon")
        {
            ph.RegisterProcess(ion_msc, particle);
            ph.RegisterProcess(new G4ionIonisation(), particle);
        }
        else if (name == "pi+" || name == "pi-")
        {
            ph.RegisterProcess(pi_msc, particle);
            ph.RegisterProcess(new G4hIonisation(), particle);
            ph.RegisterProcess(pi_brems, particle);
            ph.RegisterProcess(pi_pair, particle);
            ph.RegisterProcess(pi_coulomb, particle);
        }
        else if (name == "kaon+" || name == "kaon-")
        {
            ph.RegisterProcess(ka_msc, particle);
            ph.RegisterProcess(new G4hIonisation(), particle);
            ph.RegisterProcess(ka_brems, particle);
            ph.RegisterProcess(ka_pair, particle);
            ph.RegisterProcess(ka_coulomb, particle);
        }
        else if (name == "proton" || name == "anti_proton")
        {
            G4hMultipleScattering* prot_msc = new G4hMultipleScattering();
            prot_msc->SetEmModel(new G4WentzelVIModel());

            ph.RegisterProcess(prot_msc, particle);
            ph.RegisterProcess(new G4hIonisation(), particle);
            ph.RegisterProcess(prot_brems, particle);
            ph.RegisterProcess(prot_pair, particle);
            ph.RegisterProcess(new G4CoulombScattering(), particle);
        }
        else if (name == "B+" || name == "B-" || name == "D+" || name == "D-"
                 || name == "Ds+" || name == "Ds-" || name == "anti_He3"
                 || name == "anti_alpha" || name == "anti_deuteron"
                 || name == "anti_lambda_c+" || name == "anti_omega-"
                 || name == "anti_sigma_c+" || name == "anti_sigma_c++"
                 || name == "anti_sigma+" || name == "anti_sigma-"
                 || name == "anti_triton" || name == "anti_xi_c+"
                 || name == "anti_xi-" || name == "deuteron"
                 || name == "lambda_c+" || name == "omega-"
                 || name == "sigma_c+" || name == "sigma_c++"
                 || name == "sigma+" || name == "sigma-" || name == "tau+"
                 || name == "tau-" || name == "triton" || name == "xi_c+"
                 || name == "xi-")
        {
            ph.RegisterProcess(ion_msc, particle);
            ph.RegisterProcess(new G4hIonisation(), particle);
        }
    }
#endif
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
