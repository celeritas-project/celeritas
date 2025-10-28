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
#include "corecel/sys/ActionGroups.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/SurfaceParams.hh"
#include "geocel/VolumeParams.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/TrackInitializer.hh"
#include "celeritas/optical/surface/SurfacePhysicsParams.hh"
#include "celeritas/track/TrackFunctors.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{

using ActionGroupsT = ActionGroups<CoreParams, CoreState>;
using SPActionGroups = std::shared_ptr<ActionGroupsT>;

namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//

template<class T>
class CollectResultsAction final : public OpticalStepActionInterface,
                                   public ConcreteAction
{
  public:
    explicit CollectResultsAction(ActionId aid, T& apply)
        : ConcreteAction(aid, "collect-results", "collect test results")
        , apply_(apply)
    {
    }

    void step(CoreParams const& params, CoreStateHost& state) const final
    {
        for (auto tid : range(TrackSlotId{state.size()}))
        {
            CoreTrackView track(params.host_ref(), state.ref(), tid);
            auto sim = track.sim();
            if ((AppliesValid{}(track)
                 && sim.post_step_action()
                        == track.surface_physics().scalars().post_boundary_action)
                || (sim.status() == TrackStatus::killed
                    && sim.post_step_action()
                           == track.surface_physics()
                                  .scalars()
                                  .surface_stepping_action))
            {
                apply_(track);
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
    T& apply_;
};

class Stepper
{
  public:
    using CoreStateHost = CoreState<MemSpace::host>;

    Stepper(CoreParams const& params)
        : params_(params)
        , actions_(std::make_shared<ActionGroupsT>(*params.action_reg()))
    {
    }

    void step(CoreStateHost& state)
    {
        for (auto const& action : actions_->step())
        {
            action->step(params_, state);
        }
    }

    void run(CoreStateHost& state)
    {
        while (state.counters().num_alive > 0)
        {
            this->step(state);
        }
    }

    void print_steps() const
    {
        for (auto const& action : actions_->step())
        {
            CELER_LOG(info)
                << action->label() << ": " << action->action_id().get();
        }
    }

  private:
    CoreParams const& params_;
    SPActionGroups actions_;
};

class OpticalAux : public AuxParamsInterface
{
  public:
    using SPConstParams = std::shared_ptr<CoreParams const>;

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

class SurfacePhysicsIntegrationTest : public GeantTestBase
{
  protected:
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

    void SetUp() override
    {
        // Construct and register optical auxiliary params
        auto& aux_reg = *this->core()->aux_reg();
        optical_ = std::make_shared<OpticalAux>(this->optical_params(),
                                                aux_reg.next_id());
        aux_reg.insert(optical_);

        // Allocate auxiliary state data, including optical core state
        size_type num_track_slots = 64;
        aux_ = std::make_shared<AuxStateVec>(
            aux_reg, MemSpace::host, StreamId{0}, num_track_slots);
        CELER_ASSERT(aux_);

        // Store a pointer to the aux state vector in the optical state
        auto& state = get<CoreState<MemSpace::host>>(*aux_, optical_->aux_id());
        state.aux() = aux_;
    }

    SPConstOpticalSurfacePhysics build_optical_surface_physics() override
    {
        inp::SurfacePhysics input;

        PhysSurfaceId phys_surface{0};
        auto add_surfaces
            = [&](std::vector<TrivialInteractionMode> const& modes) {
                  CELER_EXPECT(!modes.empty());
                  input.materials.push_back(
                      std::vector<OptMatId>(modes.size() - 1, OptMatId{0}));
                  for (auto m : modes)
                  {
                      input.roughness.polished.emplace(phys_surface,
                                                       inp::NoRoughness{});
                      input.reflectivity.fresnel.emplace(
                          phys_surface, inp::FresnelReflection{});
                      input.interaction.trivial.emplace(phys_surface, m);
                      ++phys_surface;
                  }
              };

        // center-top surface

        add_surfaces({
            TrivialInteractionMode::transmit,
            this->get_interaction_mode(),
            TrivialInteractionMode::transmit,
            TrivialInteractionMode::transmit,
        });

        // Default surface

        add_surfaces({TrivialInteractionMode::absorb});

        return std::make_shared<SurfacePhysicsParams>(
            this->optical_action_reg().get(), input);
    }

    void init_tracks(CoreState<MemSpace::host>& state,
                     std::vector<TrackInitializer> const& inits) const
    {
        for (auto tid : range(TrackSlotId(inits.size())))
        {
            CoreTrackView track(
                this->optical_params()->host_ref(), state.ref(), tid);
            track = inits[tid.get()];
        }

        {
            auto& counters = state.counters();
            counters.num_pending = 0;
            counters.num_vacancies -= inits.size();
            counters.num_active = inits.size();
            counters.num_alive = inits.size();
        }
    }

    virtual TrivialInteractionMode get_interaction_mode() const
    {
        return TrivialInteractionMode::absorb;
    }

    template<class T>
    void create_collector(T& collector) const
    {
        auto& reg = *this->optical_params()->action_reg();
        reg.insert(std::make_shared<CollectResultsAction<T>>(reg.next_id(),
                                                             collector));
    }

    void run()
    {
        auto& state = get<CoreState<MemSpace::host>>(*aux_, optical_->aux_id());

        std::vector<TrackInitializer> inits;
        std::vector<real_type> points{-40, -30, -20, -10, 0, 10, 20, 30, 40};
        for (auto x : points)
        {
            inits.push_back({units::MevEnergy{3e-6},
                             Real3{0, 0, 0},
                             make_unit_vector(Real3{x, 50, 0}),
                             Real3{0, 0, 1},
                             0,
                             ImplVolumeId{0}});
        }

        this->init_tracks(state, inits);

        Stepper stepper{*this->optical_params()};
        stepper.run(state);
    }

    std::shared_ptr<OpticalAux> optical_;
    std::shared_ptr<AuxStateVec> aux_;
};

class SurfacePhysicsIntegrationBackscatterTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    struct Collector
    {
        size_type num_back_scattered{0};

        void operator()(CoreTrackView const& track)
        {
            // CELER_LOG(info) << "Track finished boundary crossing...";
            // CELER_LOG(info) << "Track position: " << track.geometry().pos();
            // CELER_LOG(info) << "Track direction: " <<
            // track.geometry().dir(); CELER_LOG(info) << "Track volume inst: "
            // << track.geometry().volume_instance_id().get(); CELER_LOG(info)
            // << "Track post step: " << track.sim().post_step_action().get();
            // CELER_LOG(info) << "Track status: " <<
            // to_cstring(track.sim().status()); CELER_LOG(info) << "Track is
            // surface crossing: " <<
            // track.surface_physics().is_crossing_boundary();

            EXPECT_EQ(TrackStatus::alive, track.sim().status());
            EXPECT_EQ(1, track.geometry().volume_instance_id().get());
            EXPECT_FALSE(track.surface_physics().is_crossing_boundary());

            num_back_scattered++;
        }
    };

    TrivialInteractionMode get_interaction_mode() const final
    {
        return TrivialInteractionMode::backscatter;
    }
};

TEST_F(SurfacePhysicsIntegrationBackscatterTest, backscatter)
{
    Collector collect{};
    this->create_collector(collect);

    this->run();

    EXPECT_EQ(9, collect.num_back_scattered);
}

class SurfacePhysicsIntegrationAbsorbTest : public SurfacePhysicsIntegrationTest
{
  public:
    struct Collector
    {
        size_type num_absorbed{0};

        void operator()(CoreTrackView const& track)
        {
            EXPECT_EQ(TrackStatus::killed, track.sim().status());
            num_absorbed++;
        }
    };

    TrivialInteractionMode get_interaction_mode() const final
    {
        return TrivialInteractionMode::absorb;
    }
};

TEST_F(SurfacePhysicsIntegrationAbsorbTest, absorb)
{
    Collector collect{};
    this->create_collector(collect);

    this->run();

    EXPECT_EQ(9, collect.num_absorbed);
}

class SurfacePhysicsIntegrationTransmitTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    struct Collector
    {
        size_type num_transmitted{0};

        void operator()(CoreTrackView const& track)
        {
            EXPECT_EQ(TrackStatus::alive, track.sim().status());
            EXPECT_EQ(2, track.geometry().volume_instance_id().get());
            EXPECT_FALSE(track.surface_physics().is_crossing_boundary());
            num_transmitted++;
        }
    };

    TrivialInteractionMode get_interaction_mode() const final
    {
        return TrivialInteractionMode::transmit;
    }
};

TEST_F(SurfacePhysicsIntegrationTransmitTest, transmit)
{
    Collector collect{};
    this->create_collector(collect);

    this->run();

    EXPECT_EQ(9, collect.num_transmitted);
}

TEST_F(SurfacePhysicsIntegrationTest, setup)
{
    // {
    //     auto& reg = *this->optical_params()->action_reg();
    //     reg.insert(std::make_shared<CollectResultsAction<BackScatterCollector>>(reg.next_id(),
    //     BackScatterCollector{}));
    // }

    {
        auto const& volume = *this->core()->volume();
        for (auto vol : range(VolumeId{volume.num_volumes()}))
        {
            CELER_LOG(info) << "Vol " << vol.get() << ": "
                            << volume.volume_labels().at(vol);
        }

        for (auto vi : range(VolumeInstanceId{volume.num_volume_instances()}))
        {
            CELER_LOG(info) << "VolInst " << vi.get() << ": "
                            << volume.volume_instance_labels().at(vi);
        }

        auto const& surface = *this->optical_params()->surface();
        for (auto s : range(SurfaceId{surface.num_surfaces()}))
        {
            CELER_LOG(info)
                << "Surface " << s.get() << ": " << surface.labels().at(s);
        }
    }

    auto& state = get<CoreState<MemSpace::host>>(*aux_, optical_->aux_id());

    std::vector<TrackInitializer> inits;
    // {
    //     {units::MevEnergy{3e-6},
    //      Real3{0, 0, 0},
    //      Real3{0, 1, 0},
    //      Real3{1, 0, 0},
    //      0,
    //      ImplVolumeId{0}},
    // };

    std::vector<real_type> points{-40, -30, -20, -10, 0, 10, 20, 30, 40};
    for (auto x : points)
    {
        inits.push_back({units::MevEnergy{3e-6},
                         Real3{0, 0, 0},
                         make_unit_vector(Real3{x, 50, 0}),
                         Real3{0, 0, 1},
                         0,
                         ImplVolumeId{0}});
    }

    for (auto tid : range(TrackSlotId(inits.size())))
    {
        CoreTrackView track(
            this->optical_params()->host_ref(), state.ref(), tid);
        track = inits[tid.get()];
    }

    {
        auto& counters = state.counters();
        counters.num_pending = 0;
        counters.num_vacancies -= inits.size();
        counters.num_active = inits.size();
        counters.num_alive = inits.size();
    }

    Stepper stepper{*this->optical_params()};
    stepper.run(state);

    // CELER_LOG(info) << " ----- DEBUG ----- ";

    // CoreTrackView track(
    //     this->optical_params()->host_ref(), state.ref(), TrackSlotId{0});
    // EXPECT_EQ(TrackSlotId{0}, track.track_slot_id());

    // CELER_LOG(info) << "Track position: " << track.geometry().pos();
    // CELER_LOG(info) << "Track volume inst: " <<
    // track.geometry().volume_instance_id().get();

    // auto actions =
    // std::make_shared<ActionGroupsT>(*this->optical_params()->action_reg());
    // for (auto const& action : actions->step())
    // {
    //     CELER_LOG(info) << action->label() << ": " <<
    //     action->action_id().get(); action->step(*this->optical_params(),
    //     state);
    // }

    // CELER_LOG(info) << "Track position: " << track.geometry().pos();
    // CELER_LOG(info) << "Track direction: " << track.geometry().dir();
    // CELER_LOG(info) << "Track volume inst: " <<
    // track.geometry().volume_instance_id().get(); CELER_LOG(info) << "Track
    // post step: " << track.sim().post_step_action().get(); CELER_LOG(info) <<
    // "Track status: " << to_cstring(track.sim().status()); CELER_LOG(info) <<
    // "Geo normal: " << track.geometry().normal(); CELER_LOG(info) << "Track
    // is surface crossing: " << track.surface_physics().is_crossing_boundary()
    // << "\n";

    // for (auto const& action : actions->step())
    // {
    //     action->step(*this->optical_params(), state);
    // }

    // CELER_LOG(info) << "Track position: " << track.geometry().pos();
    // CELER_LOG(info) << "Track direction: " << track.geometry().dir();
    // CELER_LOG(info) << "Track volume inst: " <<
    // track.geometry().volume_instance_id().get(); CELER_LOG(info) << "Track
    // post step: " << track.sim().post_step_action().get(); CELER_LOG(info) <<
    // "Track status: " << to_cstring(track.sim().status()); CELER_LOG(info) <<
    // "Track is surface crossing: " <<
    // track.surface_physics().is_crossing_boundary();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
