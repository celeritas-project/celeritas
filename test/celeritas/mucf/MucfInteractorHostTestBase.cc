//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/MucfInteractorHostTestBase.cc
//---------------------------------------------------------------------------//
#include "MucfInteractorHostTestBase.hh"

#include "celeritas/Units.hh"
#include "celeritas/inp/MucfPhysics.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Initialize MuCF-specific particle and material parameters.
 */
MucfInteractorHostBase::MucfInteractorHostBase()
{
    using constants::stable_decay_constant;
    using AtomicMassNumber = AtomicNumber;
    using units::AmuMass;
    using units::ElementaryCharge;
    using units::MevMass;
    using units::Second;
    using InvSecond = RealQuantity<UnitInverse<Second>>;

    constexpr units::MevMass amu_mev{931.5};  // Convert from AMU to MeV
    auto const scalars = inp::MucfScalars::from_default();

    // Particle masses
    // PDG, PRD 110, 030001, 2024 (https://doi.org/10.1103/PhysRevD.110.030001)
    constexpr MevMass muon_mass{105.6583755};
    constexpr MevMass protium_mass{938.272088};
    constexpr MevMass neutron_mass{939.565420};
    // Acceleron default values
    MevMass deuterium_mass{scalars.deuterium.value() * amu_mev};
    MevMass tritium_mass{scalars.tritium.value() * amu_mev};
    // CODATA 2022 (https://arxiv.org/pdf/2409.03787)
    MevMass alpha_mass{3727.379};
    MevMass he3_mass{2808.391};

    // Decay constants
    // Muon: PDG, 110, 030001, 2024
    // Tritium: NUBASE 2020, Chinese Physics C 45 030001
    // (https://iopscience.iop.org/article/10.1088/1674-1137/abddae)
    constexpr InvSecond muon_decay_constant{1 / 2.1969811e-6};
    constexpr InvSecond tritium_decay_constant{1 / 3.8879e+8};

    // ParticlesParams used by the muCF process
    ParticleParams::Input par_inp = {
        // Leptons
        {"mu_minus",
         pdg::mu_minus(),
         muon_mass,
         ElementaryCharge{-1},
         native_value_from(muon_decay_constant)},
        {"mu_plus",
         pdg::mu_plus(),
         muon_mass,
         ElementaryCharge{1},
         native_value_from(muon_decay_constant)},
        // Nuclei and ions
        {"proton",
         pdg::proton(),
         protium_mass,
         ElementaryCharge{1},
         stable_decay_constant},
        {"neutron",
         pdg::neutron(),
         neutron_mass,
         zero_quantity(),
         stable_decay_constant},
        {"deuterium",
         pdg::deuteron(),
         deuterium_mass,
         ElementaryCharge{1},
         stable_decay_constant},
        {"tritium",
         pdg::triton(),
         tritium_mass,
         ElementaryCharge{1},
         native_value_from(tritium_decay_constant)},
        {"alpha",
         pdg::alpha(),
         alpha_mass,
         ElementaryCharge{2},
         stable_decay_constant},
        {"he3", pdg::he3(), he3_mass, ElementaryCharge{2}, stable_decay_constant},
        {"muonic_alpha",
         pdg::muonic_alpha(),
         alpha_mass + muon_mass,
         ElementaryCharge{1},
         native_value_from(muon_decay_constant)},
    };
    this->set_particle_params(std::move(par_inp));

    // Material parameters for D-T fuel mixture
    // Based on mucf-box.gdml: 50% deuterium, 50% tritium gas at 300K
    MaterialParams::Input mat_inp;

    // Define isotopes
    MevEnergy dummy_binding_energy{0};
    mat_inp.isotopes = {
        {
            AtomicNumber{1},
            AtomicMassNumber{1},
            dummy_binding_energy,
            dummy_binding_energy,
            dummy_binding_energy,
            MevMass{938.272},
            Label{"protium"},
        },
        {
            AtomicNumber{1},
            AtomicMassNumber{2},
            dummy_binding_energy,
            dummy_binding_energy,
            dummy_binding_energy,
            MevMass{1875.613},
            Label{"deuterium"},
        },
        {
            AtomicNumber{1},
            AtomicMassNumber{3},
            dummy_binding_energy,
            dummy_binding_energy,
            dummy_binding_energy,
            MevMass{2808.921},
            Label{"tritium"},
        },
    };

    // Define hydrogen element with 50/50 d and t
    mat_inp.elements
        = {{AtomicNumber{1},
            AmuMass{2.515026},  // Weighted average of 50/50 d + t
            {{IsotopeId{0}, 0.0}, {IsotopeId{1}, 0.5}, {IsotopeId{2}, 0.5}},
            Label{"H_dt"}}};

    // Number density based on the mucf-box.gdml data:
    // n = (rho * N_A) / M = (0.177496197091547 * N_A) / 2.515026
    real_type num_density = {4.25e22};  // [1 / cm^3]

    // Setup dt target material
    mat_inp.materials = {
        {num_density,
         300,  // Temperature [K]
         MatterState::gas,
         {{ElementId{0}, 1.0}},
         Label{"hdt_fuel"}},
    };

    this->set_material_params(std::move(mat_inp));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
