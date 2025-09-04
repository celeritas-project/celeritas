//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PrimaryGenerator.test.cc
//---------------------------------------------------------------------------//
#include <memory>
#include <utility>

#include "corecel/Types.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxParamsRegistry.hh"
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
#include "celeritas/optical/Transporter.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
#include "celeritas/optical/gen/OffloadData.hh"
#include "celeritas/optical/gen/PrimaryGeneratorAction.hh"
#include "celeritas/phys/GeneratorRegistry.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Temporary helper class for constructing optical aux state data.
 */
class OpticalAux : public AuxParamsInterface
{
  public:
    using SPConstParams = std::shared_ptr<optical::CoreParams const>;

  public:
    OpticalAux(SPConstParams params, AuxId id) : params_(params), aux_id_(id)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(aux_id_);
    }

    AuxId aux_id() const final { return aux_id_; }
    std::string_view label() const final { return "optical-aux"; }
    UPState create_state(MemSpace m, StreamId id, size_type size) const final
    {
        if (m == MemSpace::host)
        {
            return std::make_unique<optical::CoreState<MemSpace::host>>(
                *params_, id, size);
        }
        else if (m == MemSpace::device)
        {
            return std::make_unique<optical::CoreState<MemSpace::device>>(
                *params_, id, size);
        }
        CELER_ASSERT_UNREACHABLE();
    }

  private:
    SPConstParams params_;
    AuxId aux_id_;
};

//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class LArSpherePrimaryGeneratorTest : public LArSphereBase
{
  public:
    void SetUp() override
    {
        {
            // Create primary generator action
            inp::OpticalPrimaryGenerator inp;
            inp.num_events = 1;
            inp.primaries_per_event = 65536;
            inp.energy.energy = units::MevEnergy{1e-5};
            inp.shape = inp::PointDistribution{Real3{0, 0, 0}};
            generate_ = optical::PrimaryGeneratorAction::make_and_insert(
                *this->core(), *this->optical_params(), std::move(inp));
        }
        {
            // Create optical transporter
            optical::Transporter::Input inp;
            inp.params = this->optical_params();
            transport_ = std::make_shared<optical::Transporter>(std::move(inp));
        }
        {
            // Construct and register optical auxiliary params
            auto& aux_reg = *this->core()->aux_reg();
            optical_ = std::make_shared<OpticalAux>(this->optical_params(),
                                                    aux_reg.next_id());
            aux_reg.insert(optical_);

            // Allocate auxiliary state data, including optical core state
            size_type num_track_slots = 4096;
            aux_ = std::make_shared<AuxStateVec>(
                aux_reg, MemSpace::host, StreamId{0}, num_track_slots);
            CELER_ASSERT(aux_);

            // Store a pointer to the aux state vector in the optical state
            auto& state = get<optical::CoreState<MemSpace::host>>(
                *aux_, optical_->aux_id());
            state.aux() = aux_;
        }
    }

    SPConstAction build_along_step() override
    {
        return LArSphereBase::build_along_step();
    }

    OpticalAccumStats counters() const;

  protected:
    std::shared_ptr<optical::PrimaryGeneratorAction> generate_;
    std::shared_ptr<optical::Transporter> transport_;
    std::shared_ptr<OpticalAux> optical_;
    std::shared_ptr<AuxStateVec> aux_;
};

//---------------------------------------------------------------------------//
/*!
 * Get optical counters.
 */
OpticalAccumStats LArSpherePrimaryGeneratorTest::counters() const
{
    auto& state
        = get<optical::CoreState<MemSpace::host>>(*aux_, optical_->aux_id());
    OpticalAccumStats accum = state.accum();
    accum.generators.push_back(generate_->counters(*aux_).accum);
    return accum;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LArSpherePrimaryGeneratorTest, primary_generator)
{
    // Get the optical state
    auto& state
        = get<optical::CoreState<MemSpace::host>>(*aux_, optical_->aux_id());

    // Queue primaries for one event
    generate_->queue_primaries(state);

    // Launch the optical loop
    (*transport_)(state);

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
