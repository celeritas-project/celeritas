//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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

#include "detail/Macros.hh"

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Initialize secondary allocation on construction.
 */
InteractorHostTestBase::InteractorHostTestBase()
{
    using celeritas::ParticleRecord;
    using namespace celeritas::constants;
    using namespace celeritas::units;
    constexpr auto zero   = celeritas::zero_quantity();
    auto           stable = ParticleRecord::stable_decay_constant();

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
    mat_inp.elements  = {{29, AmuMass{63.546}, Label{"Cu"}},
                         {19, AmuMass{39.0983}, Label{"K"}},
                         {8, AmuMass{15.999}, Label{"O"}},
                         {74, AmuMass{183.84}, Label{"W"}},
                         {82, AmuMass{207.2}, Label{"Pb"}}};
    mat_inp.materials = {
        {0.141 * na_avogadro,
         293.0,
         celeritas::MatterState::solid,
         {{celeritas::ElementId{0}, 1.0}},
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
         celeritas::MatterState::solid,
         {{celeritas::ElementId{0}, 1.0}},
         Label{"Cu-1.0"}},
        {1.0 * constants::na_avogadro,
         293.0,
         celeritas::MatterState::solid,
         {{celeritas::ElementId{2}, 0.5},
          {celeritas::ElementId{3}, 0.3},
          {celeritas::ElementId{4}, 0.2}},
         Label{"PbWO"}},
    };
    this->set_material_params(std::move(mat_inp));

    // Set cutoffs
    {
        CutoffParams::Input           input;
        CutoffParams::MaterialCutoffs material_cutoffs(
            material_params_->size());
        material_cutoffs[0] = {MevEnergy{0.02064384}, 0.07};
        input.materials     = this->material_params();
        input.particles     = this->particle_params();
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
 * Set particle parameters.
 */
void InteractorHostTestBase::set_material_params(MaterialParams::Input inp)
{
    CELER_EXPECT(!inp.materials.empty());

    material_params_ = std::make_shared<MaterialParams>(std::move(inp));
    ms_              = StateStore<celeritas::MaterialStateData>(
        material_params_->host_ref(), 1);
    cutoff_params_ = {};
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the incident track's material
 */
void InteractorHostTestBase::set_material(const std::string& name)
{
    CELER_EXPECT(material_params_);

    mt_view_ = std::make_shared<MaterialTrackView>(
        material_params_->host_ref(), ms_.ref(), ThreadId{0});

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
    ps_              = StateStore<celeritas::ParticleStateData>(
        particle_params_->host_ref(), 1);
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
        particle_params_->host_ref(), ps_.ref(), ThreadId{0});

    // Initialize
    ParticleTrackView::Initializer_t init;
    init.particle_id = particle_params_->find(pdg);
    init.energy      = energy;
    *pt_view_        = init;
}

//---------------------------------------------------------------------------//
/*!
 * Set an incident direction (and normalize it).
 */
void InteractorHostTestBase::set_inc_direction(const Real3& dir)
{
    CELER_EXPECT(celeritas::norm(dir) > 0);

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
void InteractorHostTestBase::check_conservation(const Interaction& interaction) const
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
    const Interaction& interaction) const
{
    // Sum of exiting kinetic energy
    real_type exit_energy = interaction.energy_deposition.value();

    // Subtract contribution from exiting particle state
    if (interaction.action != Action::absorbed)
    {
        exit_energy += interaction.energy.value();
    }

    // Subtract contributions from exiting secondaries
    for (const Secondary& s : interaction.secondaries)
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
    const Interaction& interaction) const
{
    CollectionStateStore<celeritas::ParticleStateData, celeritas::MemSpace::host>
                      temp_store(particle_params_->host_ref(), 1);
    ParticleTrackView temp_track(
        particle_params_->host_ref(), temp_store.ref(), ThreadId{0});

    const auto& parent_track = this->particle_track();

    // Sum of exiting momentum
    Real3 exit_momentum = {0, 0, 0};

    // Subtract contribution from exiting particle state
    if (interaction.action != Action::absorbed)
    {
        ParticleTrackView::Initializer_t init;
        init.particle_id = parent_track.particle_id();
        init.energy      = interaction.energy;
        temp_track       = init;
        axpy(temp_track.momentum().value(),
             interaction.direction,
             &exit_momentum);
    }

    // Subtract contributions from exiting secondaries
    for (const Secondary& s : interaction.secondaries)
    {
        ParticleTrackView::Initializer_t init;
        init.particle_id = s.particle_id;
        init.energy      = s.energy;
        temp_track       = init;
        axpy(temp_track.momentum().value(), s.direction, &exit_momentum);
    }

    // Compare against incident particle
    {
        Real3 delta_momentum = exit_momentum;
        axpy(-parent_track.momentum().value(), inc_direction_, &delta_momentum);
        EXPECT_SOFT_NEAR(0.0,
                         dot_product(delta_momentum, delta_momentum),
                         parent_track.momentum().value() * 1e-12)
            << "Incident: " << inc_direction_
            << " with p = " << parent_track.momentum().value()
            << "* MeV/c; exiting p = " << exit_momentum;
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
