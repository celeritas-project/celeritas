//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/MucfInteractorHostTestBase.cc
//---------------------------------------------------------------------------//
#include "MucfInteractorHostTestBase.hh"

#include "celeritas/Units.hh"
#include "celeritas/grid/NonuniformGridBuilder.hh"
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

        // Ions
        {"proton",
         pdg::proton(),
         protium_mass,
         ElementaryCharge{1},
         stable_decay_constant},
        {"tritium",
         pdg::triton(),
         tritium_mass,
         ElementaryCharge{1},
         native_value_from(tritium_decay_constant)},
        {"neutron",
         pdg::neutron(),
         neutron_mass,
         zero_quantity(),
         stable_decay_constant},
        {"alpha",
         pdg::alpha(),
         alpha_mass,
         ElementaryCharge{2},
         stable_decay_constant},
        {"he3", pdg::he3(), he3_mass, ElementaryCharge{2}, stable_decay_constant},
        {"deuterium",
         pdg::deuteron(),
         deuterium_mass,
         ElementaryCharge{1},
         stable_decay_constant},

        // Muonic atoms
        {"muonic_hydrogen",
         pdg::muonic_hydrogen(),
         protium_mass + muon_mass,
         ElementaryCharge{1},
         stable_decay_constant},
        {"muonic_deuteron",
         pdg::muonic_deuteron(),
         deuterium_mass + muon_mass,
         ElementaryCharge{1},
         stable_decay_constant},
        {"muonic_triton",
         pdg::muonic_triton(),
         tritium_mass + muon_mass,
         ElementaryCharge{1},
         stable_decay_constant},
        {"muonic_alpha",
         pdg::muonic_alpha(),
         alpha_mass + muon_mass,
         ElementaryCharge{2},
         native_value_from(muon_decay_constant)},
        {"muonic_he3",
         pdg::muonic_he3(),
         he3_mass + muon_mass,
         ElementaryCharge{2},
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
/*!
 * Return a populated \c DTMixMucfData host data.
 */
HostVal<DTMixMucfData> MucfInteractorHostBase::make_host_data()
{
    using AtomicMassNumber = AtomicNumber;
    using MaterialFractionsArray = EnumArray<MucfIsotope, real_type>;
    using MoleculeCycles = Array<real_type, 2>;
    using CycleTimesArray = EnumArray<MucfMuonicMolecule, MoleculeCycles>;

    auto const& particles = *this->particle_params();
    this->set_material("hdt_fuel");

    HostVal<DTMixMucfData> host_data;

    // Set up particle IDs
    host_data.particle_ids.mu_minus = particles.find(pdg::mu_minus());

    host_data.particle_ids.proton = particles.find(pdg::proton());
    host_data.particle_ids.triton = particles.find(pdg::triton());
    host_data.particle_ids.neutron = particles.find(pdg::neutron());
    host_data.particle_ids.alpha = particles.find(pdg::alpha());
    host_data.particle_ids.he3 = particles.find(pdg::he3());

    host_data.particle_ids.muonic_hydrogen
        = particles.find(pdg::muonic_hydrogen());
    host_data.particle_ids.muonic_deuteron
        = particles.find(pdg::muonic_deuteron());
    host_data.particle_ids.muonic_triton = particles.find(pdg::muonic_triton());
    host_data.particle_ids.muonic_alpha = particles.find(pdg::muonic_alpha());
    host_data.particle_ids.muonic_he3 = particles.find(pdg::muonic_he3());

    // Set up particle masses
    host_data.particle_masses.mu_minus
        = particles.get(host_data.particle_ids.mu_minus).mass();

    host_data.particle_masses.proton
        = particles.get(host_data.particle_ids.proton).mass();
    host_data.particle_masses.triton
        = particles.get(host_data.particle_ids.triton).mass();
    host_data.particle_masses.neutron
        = particles.get(host_data.particle_ids.neutron).mass();
    host_data.particle_masses.alpha
        = particles.get(host_data.particle_ids.alpha).mass();
    host_data.particle_masses.he3
        = particles.get(host_data.particle_ids.he3).mass();

    host_data.particle_masses.muonic_hydrogen
        = particles.get(host_data.particle_ids.muonic_hydrogen).mass();
    host_data.particle_masses.muonic_deuteron
        = particles.get(host_data.particle_ids.muonic_deuteron).mass();
    host_data.particle_masses.muonic_triton
        = particles.get(host_data.particle_ids.muonic_triton).mass();
    host_data.particle_masses.muonic_alpha
        = particles.get(host_data.particle_ids.muonic_alpha).mass();
    host_data.particle_masses.muonic_he3
        = particles.get(host_data.particle_ids.muonic_he3).mass();

    // Set up muon energy CDF
    auto const inp_data = inp::MucfPhysics::from_default();
    NonuniformGridBuilder build_grid_record{&host_data.reals};
    host_data.muon_energy_cdf = build_grid_record(inp_data.muon_energy_cdf);

    auto const& material = *this->material_params();
    auto const& el_view = material.get(ElementId{0});  // Only one element

    MaterialFractionsArray iso_fractions_array;
    for (auto const& frac : el_view.isotopes())
    {
        auto const& iso_view = material.get(frac.isotope);
        if (iso_view.atomic_number() != AtomicNumber{1})
        {
            // Skip non-hydrogen (if added later to test)
            continue;
        }

        // Set up isotopic fractions for D and T

        if (iso_view.atomic_mass_number() == AtomicMassNumber{2})
        {
            iso_fractions_array[MucfIsotope::deuterium] = frac.fraction;
        }
        if (iso_view.atomic_mass_number() == AtomicMassNumber{3})
        {
            iso_fractions_array[MucfIsotope::tritium] = frac.fraction;
        }
    }

    // Set up fractions
    CollectionBuilder<MaterialFractionsArray, MemSpace::host, MuCfMatId>
        host_iso_frac(&host_data.isotopic_fractions);
    host_iso_frac.push_back(std::move(iso_fractions_array));

    // Set up mucf material id to physics material id mapping
    CollectionBuilder<PhysMatId, MemSpace::host, MuCfMatId> host_matid(
        &host_data.mucfmatid_to_matid);
    auto const& mat_view = material.get(PhysMatId{0});  // Only one material
    host_matid.push_back(mat_view.material_id());

    // Set up cycle times (numbers from DTMixMucfModel test)
    CycleTimesArray ct_array;
    ct_array[MucfMuonicMolecule::deuterium_deuterium] = {1.83e-6, 1.14};
    ct_array[MucfMuonicMolecule::deuterium_tritium] = {1.018e-8, 5.098e-9};
    ct_array[MucfMuonicMolecule::deuterium_tritium] = {1.40e-6, 0};

    CollectionBuilder<CycleTimesArray, MemSpace::host, MuCfMatId> host_ct(
        &host_data.cycle_times);
    host_ct.push_back(std::move(ct_array));

    return host_data;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
