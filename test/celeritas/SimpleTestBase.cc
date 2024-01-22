//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/SimpleTestBase.cc
//---------------------------------------------------------------------------//
#include "SimpleTestBase.hh"

#include "celeritas/Quantities.hh"
#include "celeritas/em/process/ComptonProcess.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/alongstep/AlongStepNeutralAction.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/TrackInitParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
auto SimpleTestBase::build_material() -> SPConstMaterial
{
    using namespace units;

    MaterialParams::Input inp;
    inp.elements = {{AtomicNumber{13}, AmuMass{27}, {}, "Al"}};
    inp.materials = {{2.7 * constants::na_avogadro / 27,
                      293.0,
                      MatterState::solid,
                      {{ElementId{0}, 1.0}},
                      "Al"},
                     {0, 0, MatterState::unspecified, {}, "hard vacuum"}};
    return std::make_shared<MaterialParams>(std::move(inp));
}

//---------------------------------------------------------------------------//
auto SimpleTestBase::build_geomaterial() -> SPConstGeoMaterial
{
    GeoMaterialParams::Input input;
    input.geometry = this->geometry();
    input.materials = this->material();
    input.volume_to_mat = {MaterialId{0}, MaterialId{1}, MaterialId{}};
    input.volume_labels = {Label{"inner"}, Label{"world"}, Label{"[EXTERIOR]"}};
    return std::make_shared<GeoMaterialParams>(std::move(input));
}

//---------------------------------------------------------------------------//
auto SimpleTestBase::build_particle() -> SPConstParticle
{
    using namespace constants;
    using namespace ::celeritas::units;
    ParticleParams::Input defs;
    defs.push_back({"gamma",
                    pdg::gamma(),
                    zero_quantity(),
                    zero_quantity(),
                    stable_decay_constant});
    defs.push_back({"electron",
                    pdg::electron(),
                    MevMass{0.5},
                    ElementaryCharge{-1},
                    stable_decay_constant});
    return std::make_shared<ParticleParams>(std::move(defs));
}

//---------------------------------------------------------------------------//
auto SimpleTestBase::build_cutoff() -> SPConstCutoff
{
    using namespace ::celeritas::units;
    CutoffParams::Input input;
    input.materials = this->material();
    input.particles = this->particle();
    input.cutoffs = {
        {pdg::gamma(),
         {{MevEnergy{0.01}, 0.1 * millimeter},
          {MevEnergy{100}, 100 * centimeter}}},
        {pdg::electron(),
         {{MevEnergy{1000}, 1000 * centimeter},
          {MevEnergy{1000}, 1000 * centimeter}}},
    };

    return std::make_shared<CutoffParams>(std::move(input));
}

//---------------------------------------------------------------------------//
auto SimpleTestBase::build_physics() -> SPConstPhysics
{
    PhysicsParams::Input input;
    input.options.secondary_stack_factor = this->secondary_stack_factor();

    ImportProcess compton_data;
    compton_data.particle_pdg = pdg::gamma().get();
    compton_data.secondary_pdg = pdg::electron().get();
    compton_data.process_type = ImportProcessType::electromagnetic;
    compton_data.process_class = ImportProcessClass::compton;
    {
        ImportModel kn_model;
        kn_model.model_class = ImportModelClass::klein_nishina;
        kn_model.materials.resize(this->material()->size());
        for (ImportModelMaterial& imm : kn_model.materials)
        {
            imm.energy = {1e-4, 1e8};
        }
        compton_data.models.push_back(std::move(kn_model));
    }
    {
        ImportPhysicsTable lambda;
        lambda.table_type = ImportTableType::lambda;
        lambda.x_units = ImportUnits::mev;
        lambda.y_units = ImportUnits::cm_inv;
        lambda.physics_vectors = {
            {ImportPhysicsVectorType::log,
             {1e-4, 1.0},  // energy
             {1e1, 1e0}},  // lambda (detector)
            {ImportPhysicsVectorType::log,
             {1e-4, 1.0},  // energy
             {1e-10, 1e-10}},  // lambda (world)
        };
        compton_data.tables.push_back(std::move(lambda));
    }
    {
        ImportPhysicsTable lambdap;
        lambdap.table_type = ImportTableType::lambda_prim;
        lambdap.x_units = ImportUnits::mev;
        lambdap.y_units = ImportUnits::cm_mev_inv;
        lambdap.physics_vectors = {
            {ImportPhysicsVectorType::log,
             {1.0, 1e4, 1e8},  // energy
             {1e0, 1e-2, 1e-4}},  // lambda * energy (detector)
            {ImportPhysicsVectorType::log,
             {1.0, 1e4, 1e8},  // energy
             {1e-10, 1e-10, 1e-10}},  // lambda * energy (world)
        };
        compton_data.tables.push_back(std::move(lambdap));
    }

    auto process_data = std::make_shared<ImportedProcesses>(
        std::vector<ImportProcess>{std::move(compton_data)});

    input.particles = this->particle();
    input.materials = this->material();
    input.processes
        = {std::make_shared<ComptonProcess>(input.particles, process_data)};
    input.action_registry = this->action_reg().get();

    return std::make_shared<PhysicsParams>(std::move(input));
}

//---------------------------------------------------------------------------//
auto SimpleTestBase::build_sim() -> SPConstSim
{
    SimParams::Input input;
    input.particles = this->particle();
    return std::make_shared<SimParams>(input);
}

//---------------------------------------------------------------------------//
auto SimpleTestBase::build_init() -> SPConstTrackInit
{
    TrackInitParams::Input input;
    input.capacity = 4096;
    input.max_events = 4096;
    input.track_order = TrackOrder::unsorted;
    return std::make_shared<TrackInitParams>(input);
}

//---------------------------------------------------------------------------//
auto SimpleTestBase::build_along_step() -> SPConstAction
{
    auto result = std::make_shared<AlongStepNeutralAction>(
        this->action_reg()->next_id());
    this->action_reg()->insert(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
