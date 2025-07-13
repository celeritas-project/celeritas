//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalLaunchAction.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/detail/OpticalLaunchAction.hh"

#include <memory>
#include <utility>

#include "corecel/Types.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/LArSphereBase.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/ModelImporter.hh"
#include "celeritas/optical/PhysicsParams.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
#include "celeritas/optical/gen/OffloadData.hh"
#include "celeritas/optical/gen/detail/PrimaryGeneratorAction.hh"
#include "celeritas/phys/GeneratorRegistry.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class LArSphereLaunchTest : public LArSphereBase
{
  public:
    void SetUp() override
    {
        using IMC = celeritas::optical::ImportModelClass;

        auto const& core = *this->core();

        // Build optical core params
        auto optical_params = std::make_shared<optical::CoreParams>([&] {
            optical::CoreParams::Input inp;
            inp.geometry = core.geometry();
            inp.material = this->optical_material();
            inp.rng = core.rng();
            inp.surface = core.surface();
            inp.action_reg = std::make_shared<ActionRegistry>();
            inp.gen_reg = std::make_shared<GeneratorRegistry>();
            inp.max_streams = core.max_streams();
            {
                optical::PhysicsParams::Input pp_inp;
                optical::ModelImporter importer{
                    this->imported_data(), inp.material, this->material()};
                std::vector<IMC> imcs{IMC::absorption, IMC::rayleigh};
                for (auto imc : imcs)
                {
                    if (auto builder = importer(imc))
                    {
                        pp_inp.model_builders.push_back(*builder);
                    }
                }
                pp_inp.materials = inp.material;
                pp_inp.action_registry = inp.action_reg.get();
                inp.physics = std::make_shared<optical::PhysicsParams>(
                    std::move(pp_inp));
            }
            CELER_ENSURE(inp);
            return inp;
        }());

        {
            // Create primary generator action
            inp::OpticalPrimaryGenerator inp;
            inp.num_events = 1;
            inp.primaries_per_event = 65536;
            inp.energy.energy = units::MevEnergy{1e-5};
            inp.shape = inp::PointShape{0, 0, 0};
            generate_ = detail::PrimaryGeneratorAction::make_and_insert(
                core, *optical_params, std::move(inp));
        }
        {
            // Create launch action
            detail::OpticalLaunchAction::Input inp;
            inp.num_track_slots = 4096;
            inp.auto_flush = 4096;
            inp.optical_params = std::move(optical_params);
            launch_ = detail::OpticalLaunchAction::make_and_insert(
                core, std::move(inp));
        }
        {
            // Create core state and aux data
            size_type core_track_slots = 1;
            core_state_ = std::make_shared<CoreState<MemSpace::host>>(
                core, StreamId{0}, core_track_slots);
        }
    }

    SPConstAction build_along_step() override
    {
        return LArSphereBase::build_along_step();
    }

    OpticalAccumStats counters() const;

  protected:
    std::shared_ptr<detail::PrimaryGeneratorAction> generate_;
    std::shared_ptr<detail::OpticalLaunchAction> launch_;
    std::shared_ptr<CoreState<MemSpace::host>> core_state_;
};

//---------------------------------------------------------------------------//
/*!
 * Get optical counters.
 */
OpticalAccumStats LArSphereLaunchTest::counters() const
{
    CELER_EXPECT(core_state_->aux_ptr());

    auto& aux = core_state_->aux();
    auto& state
        = dynamic_cast<optical::CoreStateBase&>(aux.at(launch_->aux_id()));
    OpticalAccumStats accum = state.accum();
    accum.generators.push_back(generate_->counters(aux).accum);
    return accum;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LArSphereLaunchTest, primary_generator)
{
    // Get the optical state and pointer to the auxiliary state vector
    auto& state = get<optical::CoreState<MemSpace::host>>(core_state_->aux(),
                                                          launch_->aux_id());
    state.aux() = core_state_->aux_ptr();

    // Queue primaries for one event
    generate_->queue_primaries(state);

    // Launch the optical loop
    launch_->step(*this->core(), *core_state_);

    // Get the accumulated counters
    auto result = this->counters();

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(105163, result.steps);
        EXPECT_EQ(34, result.step_iters);
    }
    EXPECT_EQ(1, result.flushes);
    ASSERT_EQ(1, result.generators.size());

    auto const& gen = result.generators.front();
    EXPECT_EQ(0, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);
    EXPECT_EQ(65536, gen.num_generated);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
