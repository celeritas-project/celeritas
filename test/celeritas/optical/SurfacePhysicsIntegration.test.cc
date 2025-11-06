//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsIntegration.test.cc
//---------------------------------------------------------------------------//
#include <memory>

#include "corecel/cont/ArrayIO.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/random/Histogram.hh"
#include "corecel/sys/ActionGroups.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/KernelLauncher.hh"
#include "geocel/SurfaceParams.hh"
#include "geocel/VolumeParams.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/TrackInitializer.hh"
#include "celeritas/optical/Transporter.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/gen/GeneratorBase.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
#include "celeritas/optical/gen/OffloadData.hh"
#include "celeritas/optical/surface/SurfacePhysicsParams.hh"
#include "celeritas/phys/GeneratorRegistry.hh"
#include "celeritas/track/CoreStateCounters.hh"
#include "celeritas/track/TrackFunctors.hh"
#include "celeritas/track/Utils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//

struct CollectResults
{
    size_type num_absorbed{0};
    size_type num_failed{0};
    size_type num_reflected{0};
    size_type num_refracted{0};

    void reset()
    {
        num_absorbed = 0;
        num_failed = 0;
        num_reflected = 0;
        num_refracted = 0;
    }

    void operator()(CoreTrackView const& track)
    {
        if (track.sim().status() == TrackStatus::alive)
        {
            auto vol = track.geometry().volume_instance_id();
            if (vol == VolumeInstanceId{1})
            {
                num_reflected++;
                return;
            }
            else if (vol == VolumeInstanceId{2})
            {
                num_refracted++;
                return;
            }
        }
        else if (track.sim().status() == TrackStatus::killed)
        {
            num_absorbed++;
            return;
        }

        num_failed++;
    }
};

struct SurfaceTestResults
{
    std::vector<size_type> num_absorbed;
    std::vector<size_type> num_reflected;
    std::vector<size_type> num_refracted;
};

class CollectResultsAction final : public OpticalStepActionInterface,
                                   public ConcreteAction
{
  public:
    explicit CollectResultsAction(ActionId aid, CollectResults& results)
        : ConcreteAction(aid, "collect-results", "collect test results")
        , results_(results)
    {
    }

    void step(CoreParams const& params, CoreStateHost& state) const final
    {
        for (auto tid : range(TrackSlotId{state.size()}))
        {
            CoreTrackView track(params.host_ref(), state.ref(), tid);
            auto sim = track.sim();
            if (this->is_post_boundary(track)
                || this->is_absorbed_on_boundary(track))
            {
                results_(track);
                sim.status(TrackStatus::killed);
            }
        }
    }

    void step(CoreParams const&, CoreStateDevice&) const final
    {
        CELER_NOT_IMPLEMENTED("not collecting on device");
    }

    StepActionOrder order() const final { return StepActionOrder::post; }

  private:
    inline bool is_post_boundary(CoreTrackView const& track) const
    {
        return AppliesValid{}(track)
               && track.sim().post_step_action()
                      == track.surface_physics().scalars().post_boundary_action;
    }

    inline bool is_absorbed_on_boundary(CoreTrackView const& track) const
    {
        return track.sim().status() == TrackStatus::killed
               && track.sim().post_step_action()
                      == track.surface_physics().scalars().surface_stepping_action;
    }

    CollectResults& results_;
};

class TestGeneratorAction final : public GeneratorBase
{
  public:
    struct Executor
    {
        CRefPtr<CoreParamsData, MemSpace::native> params;
        RefPtr<CoreStateData, MemSpace::native> state;
        TrackInitializer init;
        CoreStateCounters counters;

        inline CELER_FUNCTION void operator()(TrackSlotId tid) const
        {
            CELER_EXPECT(params);
            CELER_EXPECT(state);
            CoreTrackView vacancy{
                *params, *state, [&] {
                    TrackSlotId idx{index_before(counters.num_vacancies,
                                                 ThreadId(tid.get()))};
                    return state->init.vacancies[idx];
                }()};

            vacancy = init;
        }

        CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
        {
            return (*this)(TrackSlotId{tid.unchecked_get()});
        }
    };

    struct Input
    {
        size_type num_photons;
    };

    static std::shared_ptr<TestGeneratorAction>
    make_and_insert(::celeritas::CoreParams const& core_params,
                    CoreParams const& params,
                    Input&& input)
    {
        CELER_EXPECT(input.num_photons > 0);
        ActionRegistry& actions = *params.action_reg();
        AuxParamsRegistry& aux = *core_params.aux_reg();
        GeneratorRegistry& gen = *params.gen_reg();
        auto result = std::make_shared<TestGeneratorAction>(
            actions.next_id(), aux.next_id(), gen.next_id(), std::move(input));
        actions.insert(result);
        aux.insert(result);
        gen.insert(result);
        return result;
    }

    TestGeneratorAction(ActionId id,
                        AuxId aux_id,
                        GeneratorId gen_id,
                        Input input)
        : GeneratorBase(id,
                        aux_id,
                        gen_id,
                        "test-generate",
                        "generate test optical photon primaries")
        , num_photons_(input.num_photons)
    {
        data_ = TrackInitializer{units::MevEnergy{3e-6},
                                 Real3{0, 50, 0},
                                 Real3{0, 1, 0},
                                 Real3{0, 0, 1},
                                 0,
                                 ImplVolumeId{0}};
    }

    void set_incident_angle(real_type angle)
    {
        real_type sin_theta = std::sin(angle);
        real_type cos_theta = std::cos(angle);

        data_.direction = Real3{sin_theta, cos_theta, 0};
        data_.position = Real3{0, 50, 0} - data_.direction;
    }

    void insert(CoreStateBase& state) const
    {
        if (auto* s = dynamic_cast<CoreStateHost*>(&state))
        {
            CELER_EXPECT(s->aux());
            auto& aux_state = this->counters(*s->aux());
            aux_state.counters.num_pending = num_photons_;
            s->counters().num_pending = num_photons_;
        }
        else
        {
            CELER_NOT_IMPLEMENTED("TestGeneratorAction on device");
        }
    }

    UPState create_state(MemSpace, StreamId, size_type) const final
    {
        return std::make_unique<GeneratorStateBase>();
    }

    void step(CoreParams const& params, CoreStateHost& state) const final
    {
        CELER_EXPECT(state.aux());

        auto const& aux_state = this->counters(*state.aux());
        size_type num_gen = min(state.counters().num_vacancies,
                                aux_state.counters.num_pending);

        if (num_gen > 0)
        {
            Executor execute{params.ptr<MemSpace::native>(),
                             state.ptr(),
                             data_,
                             state.counters()};
            launch_action(num_gen, execute);
        }

        this->update_counters(state);
    }

    void step(CoreParams const&, CoreStateDevice&) const final
    {
        CELER_NOT_IMPLEMENTED("TestGeneratorAction on device");
    }

  private:
    size_type num_photons_;
    TrackInitializer data_;
};

class SurfacePhysicsIntegrationTest : public GeantTestBase
{
  public:
    std::string_view gdml_basename() const override { return "optical-box"; }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto result = GeantTestBase::build_geant_options();
        result.optical = {};
        CELER_ENSURE(result.optical);
        return result;
    }

    GeantImportDataSelection build_import_data_selection() const override
    {
        auto result = GeantTestBase::build_import_data_selection();
        result.processes |= GeantImportDataSelection::optical;
        return result;
    }

    std::vector<IMC> select_optical_models() const override
    {
        return {IMC::absorption};
    }

    void SetUp() override {}

    virtual void setup_surface_models(inp::SurfacePhysics&) const = 0;

    SPConstOpticalSurfacePhysics build_optical_surface_physics() override
    {
        inp::SurfacePhysics input;

        this->setup_surface_models(input);

        // Default surface

        size_type phys_surface{0};
        for (auto const& mats : input.materials)
        {
            phys_surface += mats.size() + 1;
        }

        input.materials.push_back({});
        input.roughness.polished.emplace(PhysSurfaceId{phys_surface},
                                         inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(PhysSurfaceId{phys_surface},
                                           inp::FresnelReflection{});
        input.interaction.trivial.emplace(PhysSurfaceId{phys_surface},
                                          TrivialInteractionMode::absorb);

        return std::make_shared<SurfacePhysicsParams>(
            this->optical_action_reg().get(), input);
    }

    void build_state(size_type num_tracks)
    {
        auto state = std::make_shared<CoreState<MemSpace::host>>(
            *this->optical_params(), StreamId{0}, num_tracks);
        state->aux() = std::make_shared<AuxStateVec>(
            *this->core()->aux_reg(), MemSpace::host, StreamId{0}, num_tracks);
        state_ = state;
    }

    void make_collector()
    {
        auto& reg = *this->optical_params()->action_reg();
        auto collector
            = std::make_shared<CollectResultsAction>(reg.next_id(), collect_);
        reg.insert(collector);
    }

    void build_transporter()
    {
        Transporter::Input inp;
        inp.params = this->optical_params();
        transport_ = std::make_shared<Transporter>(std::move(inp));
    }

    SurfaceTestResults run(std::vector<real_type> const& angles)
    {
        TestGeneratorAction::Input inp;
        inp.num_photons = 100;
        auto generate = TestGeneratorAction::make_and_insert(
            *this->core(), *this->optical_params(), std::move(inp));

        this->make_collector();
        this->build_state(128);
        this->build_transporter();

        SurfaceTestResults results;
        for (auto angle : angles)
        {
            collect_.reset();

            generate->set_incident_angle(angle * M_PI / 180.0);

            generate->insert(*state_);

            (*transport_)(*state_);

            EXPECT_EQ(0, collect_.num_failed);
            results.num_absorbed.push_back(collect_.num_absorbed);
            results.num_reflected.push_back(collect_.num_reflected);
            results.num_refracted.push_back(collect_.num_refracted);
        }

        return results;
    }

  protected:
    std::shared_ptr<CoreState<MemSpace::host>> state_;
    std::shared_ptr<AuxStateVec> aux_;
    std::shared_ptr<Transporter> transport_;
    CollectResults collect_;
};

class SurfacePhysicsIntegrationBackscatterTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});
        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::backscatter);
    }
};

class SurfacePhysicsIntegrationAbsorbTest : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});
        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::absorb);
    }
};

class SurfacePhysicsIntegrationTransmitTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});
        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::transmit);
    }
};

class SurfacePhysicsIntegrationFresnelTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});
        input.interaction.dielectric.emplace(
            phys_surface,
            inp::DielectricInteraction::from_dielectric(
                inp::ReflectionForm::from_spike()));
    }
};

TEST_F(SurfacePhysicsIntegrationBackscatterTest, backscatter)
{
    std::vector<real_type> angles{0, 30, 60};
    auto result = this->run(angles);

    SurfaceTestResults expected;
    expected.num_reflected = {100, 100, 100};
    expected.num_refracted = {0, 0, 0};
    expected.num_absorbed = {0, 0, 0};

    EXPECT_EQ(expected.num_reflected, result.num_reflected);
    EXPECT_EQ(expected.num_refracted, result.num_refracted);
    EXPECT_EQ(expected.num_absorbed, result.num_absorbed);
}

TEST_F(SurfacePhysicsIntegrationAbsorbTest, absorb)
{
    std::vector<real_type> angles{0, 30, 60};
    auto result = this->run(angles);

    SurfaceTestResults expected;
    expected.num_refracted = {0, 0, 0};
    expected.num_reflected = {0, 0, 0};
    expected.num_absorbed = {100, 100, 100};

    EXPECT_EQ(expected.num_reflected, result.num_reflected);
    EXPECT_EQ(expected.num_refracted, result.num_refracted);
    EXPECT_EQ(expected.num_absorbed, result.num_absorbed);
}

TEST_F(SurfacePhysicsIntegrationTransmitTest, transmit)
{
    std::vector<real_type> angles{0, 30, 60};
    auto result = this->run(angles);

    SurfaceTestResults expected;
    expected.num_refracted = {100, 100, 100};
    expected.num_reflected = {0, 0, 0};
    expected.num_absorbed = {0, 0, 0};

    EXPECT_EQ(expected.num_reflected, result.num_reflected);
    EXPECT_EQ(expected.num_refracted, result.num_refracted);
    EXPECT_EQ(expected.num_absorbed, result.num_absorbed);
}

TEST_F(SurfacePhysicsIntegrationFresnelTest, fresnel)
{
    std::vector<real_type> angles{
        0,
        10,
        20,
        30,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        60,
        70,
        80,
    };

    auto result = this->run(angles);

    static unsigned int const expected_num_absorbed[] = {
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
    };
    static unsigned int const expected_num_reflected[] = {
        2u,
        0u,
        3u,
        4u,
        15u,
        11u,
        9u,
        17u,
        18u,
        34u,
        27u,
        42u,
        60u,
        100u,
        100u,
        100u,
        100u,
        100u,
    };
    static unsigned int const expected_num_refracted[] = {
        98u,
        100u,
        97u,
        96u,
        85u,
        89u,
        91u,
        83u,
        82u,
        66u,
        73u,
        58u,
        40u,
        0u,
        0u,
        0u,
        0u,
        0u,
    };

    EXPECT_VEC_EQ(expected_num_absorbed, result.num_absorbed);
    EXPECT_VEC_EQ(expected_num_reflected, result.num_reflected);
    EXPECT_VEC_EQ(expected_num_refracted, result.num_refracted);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
