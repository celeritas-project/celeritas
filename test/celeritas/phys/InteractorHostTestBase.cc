//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/InteractorHostTestBase.cc
//---------------------------------------------------------------------------//
#include "InteractorHostTestBase.hh"

#include "corecel/data/StackAllocator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Initialize secondary allocation on construction.
 */
InteractorHostTestBase::InteractorHostTestBase()
{
    using namespace constants;
    using namespace units;
    constexpr auto zero = zero_quantity();
    auto stable = ParticleRecord::stable_decay_constant();

    constexpr MevMass emass{0.5109989461};
    constexpr MevMass mumass{105.6583745};

    // Default particle params
    ParticleParams::Input par_inp = {
        {"electron", pdg::electron(), emass, ElementaryCharge{-1}, stable},
        {"positron", pdg::positron(), emass, ElementaryCharge{1}, stable},
        {"gamma", pdg::gamma(), zero, zero, stable},
        {"mu_minus", pdg::mu_minus(), mumass, ElementaryCharge{-1}, stable},
        {"mu_plus", pdg::mu_plus(), mumass, ElementaryCharge{1}, stable},
    };
    this->set_particle_params(std::move(par_inp));

    // Default material params
    MaterialParams::Input mat_inp;
    mat_inp.elements = {{AtomicNumber{29}, AmuMass{63.546}, {}, Label{"Cu"}},
                        {AtomicNumber{19}, AmuMass{39.0983}, {}, Label{"K"}},
                        {AtomicNumber{8}, AmuMass{15.999}, {}, Label{"O"}},
                        {AtomicNumber{74}, AmuMass{183.84}, {}, Label{"W"}},
                        {AtomicNumber{82}, AmuMass{207.2}, {}, Label{"Pb"}}};
    mat_inp.materials = {
        {0.141 * na_avogadro,
         293.0,
         MatterState::solid,
         {{ElementId{0}, 1.0}},
         Label{"Cu"}},
        {0.05477 * na_avogadro,
         293.15,
         MatterState::solid,
         {{ElementId{0}, 1.0}},
         Label{"Pb"}},
        {1e-5 * na_avogadro,
         293.,
         MatterState::solid,
         {{ElementId{1}, 1.0}},
         Label{"K"}},
        {1.0 * na_avogadro,
         293.0,
         MatterState::solid,
         {{ElementId{0}, 1.0}},
         Label{"Cu-1.0"}},
        {1.0 * constants::na_avogadro,
         293.0,
         MatterState::solid,
         {{ElementId{2}, 0.5}, {ElementId{3}, 0.3}, {ElementId{4}, 0.2}},
         Label{"PbWO"}},
    };
    this->set_material_params(std::move(mat_inp));

    // Set cutoffs
    {
        CutoffParams::Input input;
        CutoffParams::MaterialCutoffs material_cutoffs(
            material_params_->size());
        material_cutoffs[0] = {MevEnergy{0.02064384}, 0.07};
        input.materials = this->material_params();
        input.particles = this->particle_params();
        input.cutoffs.insert({pdg::gamma(), material_cutoffs});
        this->set_cutoff_params(input);
    }

    // Set default capacities
    this->resize_secondaries(128);
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
InteractorHostTestBase::~InteractorHostTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Helper to make dummy ImportProcess data .
 */
ImportProcess InteractorHostTestBase::make_import_process(
    PDGNumber particle,
    PDGNumber secondary,
    ImportProcessClass ipc,
    std::vector<ImportModelClass> models) const
{
    CELER_EXPECT(particle);
    CELER_EXPECT(material_params_);
    CELER_EXPECT(!models.empty());
    ImportProcess result;
    result.particle_pdg = particle.get();
    result.secondary_pdg = secondary ? secondary.get() : 0;
    result.process_type = ImportProcessType::electromagnetic;
    result.process_class = ipc;

    for (auto& mcls : models)
    {
        ImportModel m;
        m.model_class = mcls;
        m.materials.resize(material_params_->num_materials());
        for (ImportModelMaterial& imm : m.materials)
        {
            imm.energy = {0, 1e12};
        }
        CELER_ASSERT(m);
        result.models.push_back(std::move(m));
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Set particle parameters.
 */
void InteractorHostTestBase::set_material_params(MaterialParams::Input inp)
{
    CELER_EXPECT(!inp.materials.empty());

    material_params_ = std::make_shared<MaterialParams>(std::move(inp));
    ms_ = StateStore<MaterialStateData>(material_params_->host_ref(), 1);
    cutoff_params_ = {};
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the incident track's material
 */
void InteractorHostTestBase::set_material(std::string const& name)
{
    CELER_EXPECT(material_params_);

    mt_view_ = std::make_shared<MaterialTrackView>(
        material_params_->host_ref(), ms_.ref(), TrackSlotId{0});

    // Initialize
    MaterialTrackView::Initializer_t init;
    init.material_id = material_params_->find_material(name);
    CELER_VALIDATE(init.material_id, << "no material '" << name << "' exists");
    *mt_view_ = init;
}

//---------------------------------------------------------------------------//
/*!
 * Set particle parameters.
 */
void InteractorHostTestBase::set_particle_params(ParticleParams::Input inp)
{
    CELER_EXPECT(!inp.empty());
    particle_params_ = std::make_shared<ParticleParams>(std::move(inp));
    ps_ = StateStore<ParticleStateData>(particle_params_->host_ref(), 1);
    cutoff_params_ = {};
}

//---------------------------------------------------------------------------//
/*!
 * Set cutoff parameters.
 */
void InteractorHostTestBase::set_cutoff_params(CutoffParams::Input inp)
{
    CELER_EXPECT(inp.materials && inp.particles);
    cutoff_params_ = std::make_shared<CutoffParams>(std::move(inp));
}

//---------------------------------------------------------------------------//
/*!
 * Set imported processes.
 */
void InteractorHostTestBase::set_imported_processes(
    std::vector<ImportProcess> inp)
{
    CELER_EXPECT(!inp.empty());
    imported_processes_ = std::make_shared<ImportedProcesses>(std::move(inp));
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the incident particle data
 */
void InteractorHostTestBase::set_inc_particle(PDGNumber pdg, MevEnergy energy)
{
    CELER_EXPECT(particle_params_);
    CELER_EXPECT(pdg);
    CELER_EXPECT(energy >= zero_quantity());

    // Construct track view
    pt_view_ = std::make_shared<ParticleTrackView>(
        particle_params_->host_ref(), ps_.ref(), TrackSlotId{0});

    // Initialize
    ParticleTrackView::Initializer_t init;
    init.particle_id = particle_params_->find(pdg);
    init.energy = energy;
    *pt_view_ = init;
}

//---------------------------------------------------------------------------//
/*!
 * Set an incident direction (and normalize it).
 */
void InteractorHostTestBase::set_inc_direction(Real3 const& dir)
{
    CELER_EXPECT(norm(dir) > 0);

    inc_direction_ = dir;
    normalize_direction(&inc_direction_);
}

//---------------------------------------------------------------------------//
/*!
 * Resize secondaries.
 */
void InteractorHostTestBase::resize_secondaries(int count)
{
    CELER_EXPECT(count > 0);
    secondaries_ = StateStore<SecondaryStackData>(count);
    sa_view_ = std::make_shared<StackAllocator<Secondary>>(secondaries_.ref());
}

//---------------------------------------------------------------------------//
/*!
 * Check for energy and momentum conservation in the interaction.
 */
void InteractorHostTestBase::check_conservation(Interaction const& interaction) const
{
    ASSERT_NE(interaction.action, Action::failed);

    this->check_momentum_conservation(interaction);
    this->check_energy_conservation(interaction);
}

//---------------------------------------------------------------------------//
/*!
 * Check for energy conservation in the interaction.
 */
void InteractorHostTestBase::check_energy_conservation(
    Interaction const& interaction) const
{
    // Sum of exiting kinetic energy
    real_type exit_energy = interaction.energy_deposition.value();

    // Subtract contribution from exiting particle state
    if (interaction.action != Action::absorbed)
    {
        exit_energy += interaction.energy.value();
    }

    // Subtract contributions from exiting secondaries
    for (Secondary const& s : interaction.secondaries)
    {
        exit_energy += s.energy.value();

        // Account for positron production
        if (s && s.particle_id == particle_params_->find(pdg::positron()))
        {
            exit_energy
                += 2 * particle_params_->get(s.particle_id).mass().value();
        }
    }

    // Compare against incident particle
    EXPECT_SOFT_EQ(this->particle_track().energy().value(), exit_energy);
}

//---------------------------------------------------------------------------//
/*!
 * Check for momentum conservation in the interaction.
 */
void InteractorHostTestBase::check_momentum_conservation(
    Interaction const& interaction) const
{
    CollectionStateStore<ParticleStateData, MemSpace::host> temp_store(
        particle_params_->host_ref(), 1);
    ParticleTrackView temp_track(
        particle_params_->host_ref(), temp_store.ref(), TrackSlotId{0});

    auto const& parent_track = this->particle_track();

    // Sum of exiting momentum
    Real3 exit_momentum = {0, 0, 0};

    // Subtract contribution from exiting particle state
    if (interaction.action != Action::absorbed)
    {
        ParticleTrackView::Initializer_t init;
        init.particle_id = parent_track.particle_id();
        init.energy = interaction.energy;
        temp_track = init;
        axpy(temp_track.momentum().value(),
             interaction.direction,
             &exit_momentum);
    }

    // Subtract contributions from exiting secondaries
    for (Secondary const& s : interaction.secondaries)
    {
        ParticleTrackView::Initializer_t init;
        init.particle_id = s.particle_id;
        init.energy = s.energy;
        temp_track = init;
        axpy(temp_track.momentum().value(), s.direction, &exit_momentum);
    }

    // Compare against incident particle
    {
        constexpr real_type default_tol = SoftEqual{}.rel();
        real_type parent_momentum_mag = parent_track.momentum().value();
        auto exit_momentum_mag = norm(exit_momentum);

        // Roundoff for lower energy particles can affect momentum calculation
        // see RelativisticBremTest.basic_with_lpm
        // see MollerBhabhaInteractorTest.stress_test
        EXPECT_SOFT_NEAR(
            parent_momentum_mag, exit_momentum_mag, default_tol * 10000);

        normalize_direction(&exit_momentum);
        EXPECT_SOFT_NEAR(real_type(1),
                         dot_product(inc_direction_, exit_momentum),
                         3 * default_tol)
            << "Incident direction: " << inc_direction_
            << "; exiting momentum direction: " << exit_momentum;
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
