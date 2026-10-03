//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantHandBack.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantHandBack.hh"

#include <algorithm>
#include <memory>
#include <G4DynamicParticle.hh>
#include <G4LogicalVolume.hh>
#include <G4ParticleDefinition.hh>
#include <G4Track.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VTouchable.hh>
#include <G4VUserTrackInformation.hh>

#include "corecel/Config.hh"

#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Device.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/SimpleCmsTestBase.hh"
#include "celeritas/ext/GeantTrackReconstruction.hh"
#include "celeritas/ext/detail/GeantParticleUtils.hh"
#include "celeritas/ext/detail/HandBackProcessor.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/user/HandBackTestAction.hh"
#include "celeritas/user/StepCollector.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class MarkedUserInfo final : public G4VUserTrackInformation
{
};

//---------------------------------------------------------------------------//
class GeantHandBackTest : public SimpleCmsTestBase
{
  protected:
    using VecTrack = detail::HandBackProcessor::VecTrack;

    static int get_test_event_id() { return 0; }

    static void SetUpTestCase()
    {
        GeantTrackReconstruction::get_current_event_id = get_test_event_id;
    }

    static void TearDownTestCase()
    {
        GeantTrackReconstruction::get_current_event_id = nullptr;
    }

    void SetUp() override
    {
        // Build core params, which registers the boundary action
        this->core();
        auto& action_reg = *this->action_reg();
        auto boundary = action_reg.find_action("geo-boundary");
        CELER_ASSERT(boundary);
        action_reg.insert(std::make_shared<HandBackTestAction>(
            action_reg.next_id(), boundary));

        recon_ = std::make_shared<GeantTrackReconstruction>(
            detail::make_geant_particles(*this->particle()),
            GeantTrackReconstruction::make_g4step());
        recon_->init_event();
    }

    void TearDown() override
    {
        if (recon_)
        {
            recon_->clear();
        }
    }

    // Offload a Geant4 track and create the corresponding primary
    Primary make_primary()
    {
        auto* electron = detail::make_geant_particles(
            *this->particle())[this->particle()->find(pdg::electron()).get()];
        G4Track g4track(new G4DynamicParticle(electron, G4ThreeVector()),
                        0.0,
                        G4ThreeVector());
        g4track.SetTrackID(42);
        g4track.SetParentID(3);
        user_info_ = new MarkedUserInfo;
        g4track.SetUserInformation(user_info_);

        Primary result;
        result.particle_id = this->particle()->find(pdg::electron());
        result.energy = units::MevEnergy{1000};
        // Start inside the tracker so that the electron reaches the
        // calorimeter boundary after making bremsstrahlung photons, which
        // also cross it
        result.position = from_cm(Real3{0, 120, 0});
        result.direction = {0, 1, 0};
        result.event_id = EventId{0};
        result.primary_id = recon_->acquire(g4track);
        return result;
    }

    template<MemSpace M>
    VecTrack run();

    // Check the reconstructed tracks
    void check_tracks(VecTrack const& tracks) const;

    std::shared_ptr<GeantTrackReconstruction> recon_;
    G4VUserTrackInformation* user_info_{nullptr};
};

//---------------------------------------------------------------------------//
/*!
 * Transport a primary, handing back tracks crossing a boundary.
 */
template<MemSpace M>
auto GeantHandBackTest::run() -> VecTrack
{
    auto hand_back = std::make_shared<GeantHandBack>(GeantHandBack::Input{},
                                                     /* num_streams = */ 1);
    auto processor = hand_back->make_local_processor(StreamId{0}, recon_);
    EXPECT_EQ(processor, hand_back->make_local_processor(StreamId{0}, recon_));
    auto collector = StepCollector::make_and_insert(*this->core(), {hand_back});

    StepperInput input;
    input.params = this->core();
    input.stream_id = StreamId{0};
    input.num_track_slots = 64;
    input.actions = std::make_shared<ActionSequence>(
        *this->action_reg(), ActionSequence::Options{});
    if constexpr (M == MemSpace::device)
    {
        device().create_streams(1);
    }
    Stepper<M> step(input);

    // Warmup hands nothing back
    step.warm_up();
    EXPECT_FALSE(processor->has_pending_steps());
    EXPECT_EQ(0, processor->num_tracks());

    VecTrack result;
    auto complete_step = [&] {
        auto counts = step.get();
        // Tracks are reconstructed only after the step completes
        EXPECT_TRUE(processor->has_pending_steps());
        processor->process_pending_steps();
        EXPECT_FALSE(processor->has_pending_steps());
        for (auto& t : processor->exchange_tracks())
        {
            result.push_back(std::move(t));
        }
        EXPECT_EQ(0, processor->num_tracks());
        return counts;
    };

    auto primary = this->make_primary();
    step.async({&primary, 1});
    auto counts = complete_step();
    for (int num_steps = 0; counts; ++num_steps)
    {
        EXPECT_LT(num_steps, 1024);
        if (num_steps >= 1024)
        {
            break;
        }
        step.async();
        counts = complete_step();
    }
    return result;
}

//---------------------------------------------------------------------------//
void GeantHandBackTest::check_tracks(VecTrack const& tracks) const
{
    ASSERT_FALSE(tracks.empty());

    auto const g4particles = detail::make_geant_particles(*this->particle());

    // Celeritas track ID of the offloaded track
    TrackId primary_track;
    for (auto const& hb : tracks)
    {
        if (hb.origin == TrackOrigin::offloaded)
        {
            primary_track = hb.celer_track_id;
        }
    }

    int num_offloaded{0};
    int num_children{0};
    for (auto const& hb : tracks)
    {
        ASSERT_TRUE(hb.track);
        G4Track const& track = *hb.track;
        EXPECT_EQ(HandBackReason::user, hb.reason);
        EXPECT_TRUE(hb.celer_track_id);
        EXPECT_GT(track.GetKineticEnergy(), 0);
        // Direction is reconstructed from native (possibly single) precision
        EXPECT_SOFT_NEAR(1.0, track.GetMomentumDirection().mag(), coarse_eps);
        EXPECT_GT(track.GetGlobalTime(), 0);
        EXPECT_EQ(track.GetPosition(), track.GetVertexPosition());
        EXPECT_EQ(fAlive, track.GetTrackStatus());

        // Touchable is reconstructed from the post-step volume
        auto const* touchable = track.GetTouchable();
        ASSERT_TRUE(touchable);
        ASSERT_TRUE(touchable->GetVolume());
        EXPECT_EQ(touchable->GetVolume()->GetLogicalVolume(),
                  track.GetLogicalVolumeAtVertex());

        if (hb.origin == TrackOrigin::offloaded)
        {
            // The offloaded track keeps its Geant4 identity
            ++num_offloaded;
            EXPECT_FALSE(hb.celer_parent_id);
            EXPECT_EQ(42, track.GetTrackID());
            EXPECT_EQ(3, track.GetParentID());
            EXPECT_EQ(user_info_, track.GetUserInformation());
            EXPECT_EQ(
                g4particles[this->particle()->find(pdg::electron()).get()],
                track.GetParticleDefinition());
        }
        else
        {
            // Celeritas secondaries have their own Geant4 identity
            EXPECT_TRUE(hb.celer_parent_id);
            EXPECT_EQ(
                GeantTrackReconstruction::geant_track_id(hb.celer_track_id),
                track.GetTrackID());
            if (hb.celer_parent_id == primary_track)
            {
                ++num_children;
                EXPECT_EQ(42, track.GetParentID());
            }
            else
            {
                EXPECT_EQ(GeantTrackReconstruction::geant_track_id(
                              hb.celer_parent_id),
                          track.GetParentID());
            }
            EXPECT_EQ(&recon_->placeholder_process(),
                      track.GetCreatorProcess());
            EXPECT_EQ(nullptr, track.GetUserInformation());
        }
    }
    // The primary is handed back when it first crosses the boundary
    EXPECT_EQ(1, num_offloaded);
    EXPECT_GT(num_children, 0);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GeantHandBackTest, host)
{
    auto tracks = this->run<MemSpace::host>();
    this->check_tracks(tracks);

    // The handed-back offloaded track owns its user information: dropping it
    // deletes the information, and the reconstruction no longer refers to it
    auto iter = std::find_if(tracks.begin(), tracks.end(), [](auto const& t) {
        return t.origin == TrackOrigin::offloaded;
    });
    ASSERT_NE(iter, tracks.end());
    EXPECT_EQ(user_info_, iter->track->GetUserInformation());
    tracks.clear();
}

TEST_F(GeantHandBackTest, TEST_IF_CELER_DEVICE(device))
{
    auto tracks = this->run<MemSpace::device>();
    this->check_tracks(tracks);
}

TEST_F(GeantHandBackTest, selection)
{
    GeantHandBack hand_back(GeantHandBack::Input{}, 1);
    EXPECT_TRUE(hand_back.filters().hand_back);
    EXPECT_TRUE(hand_back.filters().detectors.empty());

    auto sel = hand_back.selection();
    EXPECT_TRUE(sel.points[StepPoint::post].volume_instance_ids);
    EXPECT_FALSE(sel.points[StepPoint::pre]);
    EXPECT_TRUE(sel.hand_back_reason);
    EXPECT_TRUE(sel.parent_id && sel.primary_id && sel.particle_id);

    GeantHandBack::Input inp;
    inp.locate_touchable = false;
    GeantHandBack no_touchable(inp, 1);
    EXPECT_FALSE(
        no_touchable.selection().points[StepPoint::post].volume_instance_ids);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
