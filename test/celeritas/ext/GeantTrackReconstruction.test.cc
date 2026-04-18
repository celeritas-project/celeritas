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

class GeantTrackReconstructionTest : public ::celeritas::test::SimpleCmsTestBase
{
  protected:
    using VecParticle = GeantTrackReconstruction::VecParticle;
    using size_type = ::celeritas::size_type;

    void SetUp() override
    {
        // Load particles from Geant4
        this->physics();

        auto& table = *G4ParticleTable::GetParticleTable();
        for (auto p : {pdg::gamma(), pdg::electron(), pdg::positron()})
        {
            particles_.push_back(table.FindParticle(p.get()));
        }

        step_ = std::make_shared<G4Step>();
        step_->NewSecondaryVector();
    }

    VecParticle particles_;
    std::shared_ptr<G4Step> step_;
};

//---------------------------------------------------------------------------//

TEST_F(GeantTrackReconstructionTest, construction)
{
    // Create an empty processor first to test basic construction
    GeantTrackReconstruction recon({}, step_);

    // Test that end_event works
    recon.clear();
}

//---------------------------------------------------------------------------//

TEST_F(GeantTrackReconstructionTest, primary_registration)
{
    GeantTrackReconstruction recon(particles_, step_);

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
    PrimaryId primary_id = recon.acquire(*primary_track, ParticleId{0});

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

    PrimaryId primary_id2 = recon.acquire(*primary_track2, ParticleId{1});
    EXPECT_EQ(1, primary_id2.unchecked_get());
}

//---------------------------------------------------------------------------//

TEST_F(GeantTrackReconstructionTest, track_restoration)
{
    GeantTrackReconstruction recon(particles_, step_);

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

    PrimaryId primary_id = recon.acquire(*primary_track, ParticleId{1});

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
}

//---------------------------------------------------------------------------//

TEST_F(GeantTrackReconstructionTest, track_restoration_without_primary)
{
    GeantTrackReconstruction recon(particles_, step_);

    // Restore track without primary information (invalid PrimaryId)
    G4Track& restored_track = recon.view(ParticleId{0}, PrimaryId{});

    // Verify basic track properties
    EXPECT_EQ(particles_[0], restored_track.GetParticleDefinition());
    EXPECT_EQ(0, restored_track.GetTrackID());
    EXPECT_EQ(0, restored_track.GetParentID());
    EXPECT_EQ(nullptr, restored_track.GetUserInformation());
    EXPECT_EQ(nullptr, restored_track.GetCreatorProcess());
}

//---------------------------------------------------------------------------//

TEST_F(GeantTrackReconstructionTest, end_event_cleanup)
{
    GeantTrackReconstruction recon(particles_, step_);

    // Register some primaries
    auto primary_track1 = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[0], G4ThreeVector(1, 0, 0)),
        0.0,
        G4ThreeVector(0, 0, 0));
    primary_track1->SetTrackID(100);
    auto user_info1 = std::make_unique<MockUserTrackInformation>(10);
    primary_track1->SetUserInformation(user_info1.release());

    // Add different process pointers to test multiple process handling
    auto mock_process1 = std::make_unique<MockProcess>("TestProcess1");
    primary_track1->SetCreatorProcess(mock_process1.get());

    auto primary_track2 = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[1], G4ThreeVector(0, 1, 0)),
        0.0,
        G4ThreeVector(0, 0, 0));
    primary_track2->SetTrackID(200);
    auto user_info2 = std::make_unique<MockUserTrackInformation>(20);
    primary_track2->SetUserInformation(user_info2.release());

    auto mock_process2 = std::make_unique<MockProcess>("TestProcess2");
    primary_track2->SetCreatorProcess(mock_process2.get());

    PrimaryId id1 = recon.acquire(*primary_track1, ParticleId{0});
    PrimaryId id2 = recon.acquire(*primary_track2, ParticleId{1});

    // Verify primaries are registered
    EXPECT_EQ(0, id1.unchecked_get());
    EXPECT_EQ(1, id2.unchecked_get());

    // Restore tracks to verify data exists
    G4Track& track1 = recon.view(ParticleId{0}, id1);
    G4Track& track2 = recon.view(ParticleId{1}, id2);
    EXPECT_EQ(100, track1.GetTrackID());
    EXPECT_EQ(200, track2.GetTrackID());

    // Verify that different process pointers are correctly restored
    EXPECT_EQ(mock_process1.get(), track1.GetCreatorProcess());
    EXPECT_EQ(mock_process2.get(), track2.GetCreatorProcess());
    EXPECT_NE(track1.GetCreatorProcess(), track2.GetCreatorProcess());

    // End event should clear reconstruction data
    recon.clear();

    // Verify all tracks have cleared user information
    for (auto particle_id :
         range(ParticleId{static_cast<size_type>(particles_.size())}))
    {
        G4Track& track = recon.view(particle_id, PrimaryId{});
        EXPECT_EQ(nullptr, track.GetUserInformation());
    }
}

//---------------------------------------------------------------------------//

TEST_F(GeantTrackReconstructionTest, multiple_particle_types)
{
    GeantTrackReconstruction recon(particles_, step_);

    // Test all particle types can be restored
    for (auto i : range(particles_.size()))
    {
        ParticleId particle_id{static_cast<size_type>(i)};
        G4Track& track = recon.view(particle_id, PrimaryId{});

        EXPECT_EQ(particles_[i], track.GetParticleDefinition());
        EXPECT_EQ(0, track.GetTrackID());
        EXPECT_EQ(0, track.GetParentID());
    }
}

//---------------------------------------------------------------------------//

TEST_F(GeantTrackReconstructionTest, reconstruction_data_persistence)
{
    GeantTrackReconstruction recon(particles_, step_);

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

    PrimaryId primary_id = recon.acquire(*primary_track, ParticleId{2});

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
}

//---------------------------------------------------------------------------//
/*!
 * Verify view() uses flush-local indexing across multiple clear() cycles.
 *
 * Simulates two Flush() calls within one event (e.g. auto_flush_ triggered
 * mid-event). After clear(), primary IDs reset to 0 for the next flush so
 * that they always index directly into the current g4_track_data_ vector.
 */
TEST_F(GeantTrackReconstructionTest, multi_flush_view)
{
    GeantTrackReconstruction recon(particles_, step_);

    // --- Flush 1: acquire one primary (flush-local id = 0) ---
    auto track1 = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[0], G4ThreeVector(1, 0, 0)),
        0.0,
        G4ThreeVector());
    track1->SetTrackID(111);
    auto mock_proc1 = std::make_unique<MockProcess>("proc1");
    track1->SetCreatorProcess(mock_proc1.get());

    PrimaryId id1 = recon.acquire(*track1, ParticleId{0});
    EXPECT_EQ(0, id1.unchecked_get());

    // view() must find track1 via id1
    EXPECT_EQ(111, recon.view(ParticleId{0}, id1).GetTrackID());

    // Simulate end of first flush
    recon.clear();

    // --- Flush 2: acquire one primary (flush-local id = 0 again) ---
    auto track2 = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[1], G4ThreeVector(0, 1, 0)),
        0.0,
        G4ThreeVector());
    track2->SetTrackID(222);
    auto mock_proc2 = std::make_unique<MockProcess>("proc2");
    track2->SetCreatorProcess(mock_proc2.get());

    PrimaryId id2 = recon.acquire(*track2, ParticleId{1});
    EXPECT_EQ(0, id2.unchecked_get());

    // view() must find track2 via id2
    EXPECT_EQ(222, recon.view(ParticleId{1}, id2).GetTrackID());
}

//---------------------------------------------------------------------------//
/*!
 * Verify view_initial restores handover-time kinematics and primary particle.
 */
TEST_F(GeantTrackReconstructionTest, view_initial)
{
    GeantTrackReconstruction recon(particles_, step_);

    // Create primary with known kinematics
    auto primary_track = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[1], G4ThreeVector(0, 0, 1), 500.0),
        3.14,
        G4ThreeVector(10, 20, 30));
    primary_track->SetTrackID(42);
    primary_track->SetParentID(0);

    PrimaryId pid = recon.acquire(*primary_track, ParticleId{1});

    // Modify the track to simulate post-transport state
    G4Track& track = recon.view(ParticleId{1}, pid);
    track.SetPosition(G4ThreeVector(99, 99, 99));
    track.SetKineticEnergy(0.0);

    // view_initial should restore original handover state
    G4Track& initial = recon.view_initial(ParticleId{1}, pid);
    EXPECT_EQ(42, initial.GetTrackID());
    EXPECT_EQ(0, initial.GetParentID());
    EXPECT_DOUBLE_EQ(500.0, initial.GetKineticEnergy());
    EXPECT_DOUBLE_EQ(3.14, initial.GetGlobalTime());
    EXPECT_DOUBLE_EQ(10, initial.GetPosition().x());
    EXPECT_DOUBLE_EQ(20, initial.GetPosition().y());
    EXPECT_DOUBLE_EQ(30, initial.GetPosition().z());
    EXPECT_DOUBLE_EQ(0, initial.GetMomentumDirection().x());
    EXPECT_DOUBLE_EQ(0, initial.GetMomentumDirection().y());
    EXPECT_DOUBLE_EQ(1, initial.GetMomentumDirection().z());
}

//---------------------------------------------------------------------------//
/*!
 * Verify is_generator_primary distinguishes generator primaries from
 * re-offloaded secondaries.
 */
TEST_F(GeantTrackReconstructionTest, is_generator_primary)
{
    GeantTrackReconstruction recon(particles_, step_);

    // Generator primary (parent_id == 0)
    auto gen_track = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[0], G4ThreeVector(1, 0, 0)),
        0.0,
        G4ThreeVector());
    gen_track->SetTrackID(1);
    gen_track->SetParentID(0);
    PrimaryId gen_id = recon.acquire(*gen_track, ParticleId{0});

    // Re-offloaded secondary (parent_id != 0)
    auto sec_track = std::make_unique<G4Track>(
        new G4DynamicParticle(particles_[1], G4ThreeVector(0, 1, 0)),
        0.0,
        G4ThreeVector());
    sec_track->SetTrackID(2);
    sec_track->SetParentID(1);
    PrimaryId sec_id = recon.acquire(*sec_track, ParticleId{1});

    EXPECT_TRUE(recon.is_generator_primary(gen_id));
    EXPECT_FALSE(recon.is_generator_primary(sec_id));
}

//---------------------------------------------------------------------------//
/*!
 * Verify for_each_primary iterates all acquired primaries with restored state.
 */
TEST_F(GeantTrackReconstructionTest, for_each_primary)
{
    GeantTrackReconstruction recon(particles_, step_);

    // Register 5 primaries cycling through particle types (gamma, e-, e+)
    int const expected_ids[] = {10, 20, 30, 40, 50};
    std::vector<std::unique_ptr<G4Track>> src_tracks;
    for (size_type i = 0; i < 5; ++i)
    {
        auto pidx = i % particles_.size();
        src_tracks.push_back(std::make_unique<G4Track>(
            new G4DynamicParticle(particles_[pidx], G4ThreeVector()),
            0.0,
            G4ThreeVector()));
        src_tracks.back()->SetTrackID(expected_ids[i]);
        PrimaryId pid = recon.acquire(
            *src_tracks.back(), ParticleId{static_cast<size_type>(pidx)});
        EXPECT_EQ(i, pid.unchecked_get());
    }

    std::vector<int> visited_ids;
    recon.for_each_primary([&visited_ids](G4Track& track) {
        visited_ids.push_back(track.GetTrackID());
    });

    ASSERT_EQ(5, visited_ids.size());
    for (size_type i = 0; i < 5; ++i)
    {
        EXPECT_EQ(expected_ids[i], visited_ids[i]);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
