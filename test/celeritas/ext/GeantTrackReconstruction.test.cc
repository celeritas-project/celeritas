//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackReconstruction.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantTrackReconstruction.hh"

#include <G4DynamicParticle.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4ProcessType.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4Track.hh>
#include <G4VProcess.hh>
#include <G4VUserTrackInformation.hh>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "celeritas/SimpleCmsTestBase.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class MockUserTrackInformation : public G4VUserTrackInformation
{
  public:
    explicit MockUserTrackInformation(int value) : value_(value) {}
    int value() const { return value_; }

  private:
    int value_;
};

// Simple mock pointer class to test process pointer storage/restoration
class MockProcess : public G4VProcess
{
  public:
    explicit MockProcess(std::string name) : G4VProcess(name) {}
    G4VParticleChange* PostStepDoIt(G4Track const&, G4Step const&) override
    {
        return nullptr;
    }

    G4VParticleChange* AlongStepDoIt(G4Track const&, G4Step const&) override
    {
        return nullptr;
    }
    G4VParticleChange* AtRestDoIt(G4Track const&, G4Step const&) override
    {
        return nullptr;
    }
    G4double AlongStepGetPhysicalInteractionLength(G4Track const&,
                                                   G4double,
                                                   G4double,
                                                   G4double&,
                                                   G4GPILSelection*) override
    {
        return 0.0;
    }

    G4double AtRestGetPhysicalInteractionLength(G4Track const&,
                                                G4ForceCondition*) override
    {
        return 0.0;
    }

    G4double PostStepGetPhysicalInteractionLength(G4Track const&,
                                                  G4double,
                                                  G4ForceCondition*) override
    {
        return 0.0;
    }
};

//---------------------------------------------------------------------------//

class GtrTest : public ::celeritas::test::SimpleCmsTestBase
{
  protected:
    using VecParticle = GeantTrackReconstruction::VecParticle;
    using size_type = ::celeritas::size_type;

    // Mocking setup for replacement G4EventManager
    static int test_cur_event;
    static int get_test_current_event_id() { return test_cur_event; }

    static void SetUpTestCase()
    {
        // Set up event ID checking (only used in CELERITAS_DEBUG) to point to
        // our test harness's current event
        GeantTrackReconstruction::get_current_event_id
            = get_test_current_event_id;
    }

    static void TearDownTestCase()
    {
        GeantTrackReconstruction::get_current_event_id = nullptr;
    }

    void SetUp() override
    {
        // Load particles from Geant4
        this->physics();

        auto& table = *G4ParticleTable::GetParticleTable();
        for (auto p : {pdg::gamma(), pdg::electron(), pdg::positron()})
        {
            particles_.push_back(table.FindParticle(p.get()));
        }

        // Create step and check it
        step_ = GeantTrackReconstruction::make_g4step();
        ASSERT_TRUE(step_);
        EXPECT_TRUE(step_->GetSecondary());

        // Reset event ID
        GtrTest::test_cur_event = 0;
    }

    VecParticle particles_;
    std::shared_ptr<G4Step> step_;
};

int GtrTest::test_cur_event{0};

//---------------------------------------------------------------------------//

TEST_F(GtrTest, TEST_IF_CELERITAS_DEBUG(empty_construction))
{
    // Create an empty processor first to test basic construction
    EXPECT_THROW(GeantTrackReconstruction({}, step_), DebugError);
}

//---------------------------------------------------------------------------//

TEST_F(GtrTest, primary_registration)
{
    GeantTrackReconstruction recon(particles_, step_);

    GtrTest::test_cur_event = 1;
    recon.init_event();

    // Create a primary track
    auto primary_track = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[0], G4ThreeVector(1, 0, 0)),
        0.0,
        G4ThreeVector(0, 0, 0));
    primary_track->SetTrackID(123);
    primary_track->SetParentID(0);

    // Add user information
    auto user_info = std::make_unique<MockUserTrackInformation>(42);
    primary_track->SetUserInformation(user_info.release());

    // Set creator process using mock process pointer
    auto mock_process = std::make_unique<MockProcess>("TestCompton");
    primary_track->SetCreatorProcess(mock_process.get());

    // Register primary
    PrimaryId primary_id = recon.acquire(*primary_track);

    // Verify primary ID
    EXPECT_EQ(0, primary_id.unchecked_get());

    // Verify user information was taken from the primary track
    EXPECT_EQ(nullptr, primary_track->GetUserInformation());

    // Test that process information can be retrieved by restoring the track
    G4Track& test_restored = recon.view(ParticleId{0}, primary_id);
    EXPECT_EQ(mock_process.get(), test_restored.GetCreatorProcess());
    EXPECT_EQ(123, test_restored.GetTrackID());

    // Register another primary
    auto primary_track2 = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[1], G4ThreeVector(0, 1, 0)),
        0.0,
        G4ThreeVector(1, 1, 1));
    primary_track2->SetTrackID(456);
    primary_track2->SetParentID(0);

    PrimaryId primary_id2 = recon.acquire(*primary_track2);
    EXPECT_EQ(1, primary_id2.unchecked_get());

    recon.clear();
}

//---------------------------------------------------------------------------//

TEST_F(GtrTest, track_restoration)
{
    GeantTrackReconstruction recon(particles_, step_);

    recon.init_event();

    // Create and register primary track with user information
    auto primary_track = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[1], G4ThreeVector(0, 0, 1)),
        0.0,
        G4ThreeVector(0, 0, 0));
    primary_track->SetTrackID(789);
    primary_track->SetParentID(1);

    auto user_info = std::make_unique<MockUserTrackInformation>(99);
    primary_track->SetUserInformation(user_info.release());

    // Set creator process using mock process pointer
    auto mock_process = std::make_unique<MockProcess>("TestBremsstrahlung");
    primary_track->SetCreatorProcess(mock_process.get());

    PrimaryId primary_id = recon.acquire(*primary_track);

    // Restore track for electron (particle ID 1) with primary information
    G4Track& restored_track = recon.view(ParticleId{1}, primary_id);

    // Verify restored track properties
    EXPECT_EQ(789, restored_track.GetTrackID());
    EXPECT_EQ(1, restored_track.GetParentID());
    EXPECT_EQ(mock_process.get(), restored_track.GetCreatorProcess());
    EXPECT_EQ(step_.get(), restored_track.GetStep());

    // Verify user information was restored
    auto* restored_user_info = dynamic_cast<MockUserTrackInformation*>(
        restored_track.GetUserInformation());
    ASSERT_NE(nullptr, restored_user_info);
    EXPECT_EQ(99, restored_user_info->value());

    // Verify particle type
    EXPECT_EQ(particles_[1], restored_track.GetParticleDefinition());

    // Clear after retrieving all tracks
    recon.clear();
}

//---------------------------------------------------------------------------//

TEST_F(GtrTest, track_restoration_without_primary)
{
    GeantTrackReconstruction recon(particles_, step_);

    recon.init_event();

    // Restore track without primary information (invalid PrimaryId)
    G4Track& restored_track = recon.view(ParticleId{0});

    // Verify basic track properties
    EXPECT_EQ(particles_[0], restored_track.GetParticleDefinition());
    EXPECT_EQ(0, restored_track.GetTrackID());
    EXPECT_EQ(0, restored_track.GetParentID());
    EXPECT_EQ(nullptr, restored_track.GetUserInformation());
    EXPECT_EQ(nullptr, restored_track.GetCreatorProcess());
}

//---------------------------------------------------------------------------//

TEST_F(GtrTest, end_event_cleanup)
{
    GeantTrackReconstruction recon(particles_, step_);

    recon.init_event();

    constexpr size_type num_primaries{2};

    Array<G4ThreeVector, num_primaries> const directions{
        {G4ThreeVector(1, 0, 0), G4ThreeVector(0, 1, 0)}};

    Array<std::unique_ptr<G4Track>, num_primaries> primaries;
    Array<std::unique_ptr<MockProcess>, num_primaries> processes;

    for (auto i : range(num_primaries))
    {
        processes[i]
            = std::make_unique<MockProcess>("MockProcess" + std::to_string(i));
    }
    EXPECT_NE(processes[0].get(), processes[1].get());

    for (auto event : range(1, 3))
    {
        SCOPED_TRACE("event" + std::to_string(event));
        recon.init_event();
        for (auto flush : range(3))
        {
            // Simulate G4 loop: create track and acquire
            Array<PrimaryId, num_primaries> primary_ids;
            for (auto i : range(num_primaries))
            {
                // Initialize track
                auto track = std::make_unique<G4Track>(
                    new G4DynamicParticle(particles_[i], directions[i]),
                    /* time = */ 0.0,
                    /* position = */ G4ThreeVector(0, 0, 0));
                track->SetTrackID(flush * 100 + i);
                auto user_info
                    = std::make_unique<MockUserTrackInformation>(10 * i);
                track->SetUserInformation(user_info.release());
                track->SetCreatorProcess(processes[i].get());

                primary_ids[i] = recon.acquire(*track);
                EXPECT_EQ(i + flush * num_primaries,
                          primary_ids[i].unchecked_get());
            }

            // Simulate celeritas loop: acquire
            for (auto i : range(num_primaries))
            {
                G4Track& track = recon.view(ParticleId{i}, primary_ids[i]);
                EXPECT_EQ(flush * 100 + i, track.GetTrackID());
                EXPECT_EQ(processes[i].get(), track.GetCreatorProcess());
            }

            if constexpr (CELERITAS_DEBUG)
            {
                // Check that we can't restore a particle from a previous event
                GtrTest::test_cur_event++;
                EXPECT_THROW(static_cast<void>(
                                 recon.view(ParticleId{0}, primary_ids[0])),
                             RuntimeError);

                // Check that we can't push a particle from a new event
                auto track = std::make_unique<G4Track>(
                    new G4DynamicParticle(particles_[0], directions[0]),
                    10.0,
                    G4ThreeVector(1, 0, 0));
                EXPECT_THROW(static_cast<void>(recon.acquire(*track)),
                             RuntimeError);

                --GtrTest::test_cur_event;
            }

            // Flush should clear reconstruction data
            recon.clear();

            // Verify all tracks have cleared user information
            for (auto particle_id :
                 range(ParticleId{static_cast<size_type>(particles_.size())}))
            {
                G4Track& track = recon.view(particle_id);
                EXPECT_EQ(nullptr, track.GetUserInformation());
            }
        }
    }
}

//---------------------------------------------------------------------------//

TEST_F(GtrTest, multiple_particle_types)
{
    GeantTrackReconstruction recon(particles_, step_);
    recon.init_event();

    // Test all particle types can be restored
    for (auto i : range(particles_.size()))
    {
        ParticleId particle_id{static_cast<size_type>(i)};
        G4Track& track = recon.view(particle_id);

        EXPECT_EQ(particles_[i], track.GetParticleDefinition());
        EXPECT_EQ(0, track.GetTrackID());
        EXPECT_EQ(0, track.GetParentID());
    }
}

//---------------------------------------------------------------------------//

TEST_F(GtrTest, reconstruction_data_persistence)
{
    GeantTrackReconstruction recon(particles_, step_);
    recon.init_event();

    // Create primary with complete information
    auto primary_track = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[2], G4ThreeVector(1, 1, 1)),
        0.0,
        G4ThreeVector(10, 20, 30));
    primary_track->SetTrackID(999);
    primary_track->SetParentID(1);

    auto user_info = std::make_unique<MockUserTrackInformation>(777);
    primary_track->SetUserInformation(user_info.release());

    // Set creator process using mock process pointer
    auto mock_process = std::make_unique<MockProcess>("TestIonization");
    primary_track->SetCreatorProcess(mock_process.get());

    PrimaryId primary_id = recon.acquire(*primary_track);

    // Test reconstruction data persists across multiple restore calls
    for (int i = 0; i < 3; ++i)
    {
        G4Track& restored = recon.view(ParticleId{2}, primary_id);

        EXPECT_EQ(999, restored.GetTrackID());
        EXPECT_EQ(1, restored.GetParentID());
        EXPECT_EQ(mock_process.get(), restored.GetCreatorProcess());

        auto* restored_info = dynamic_cast<MockUserTrackInformation*>(
            restored.GetUserInformation());
        ASSERT_NE(nullptr, restored_info);
        EXPECT_EQ(777, restored_info->value());
    }

    // Clear after retrieving all tracks
    recon.clear();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
