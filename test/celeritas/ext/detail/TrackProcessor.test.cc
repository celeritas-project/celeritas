//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/TrackProcessor.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/detail/TrackProcessor.hh"

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
namespace detail
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

class TrackProcessorTest : public ::celeritas::test::SimpleCmsTestBase
{
  protected:
    using VecParticle = TrackProcessor::VecParticle;
    using size_type = ::celeritas::size_type;

    VecParticle make_particles()
    {
        // Load particles from Geant4
        this->physics();

        VecParticle result;
        auto& table = *G4ParticleTable::GetParticleTable();
        for (auto p : {pdg::gamma(), pdg::electron(), pdg::positron()})
        {
            result.push_back(table.FindParticle(p.get()));
        }
        return result;
    }
};

//---------------------------------------------------------------------------//

TEST_F(TrackProcessorTest, construction)
{
    // Create an empty processor first to test basic construction
    TrackProcessor::VecParticle empty_particles;
    TrackProcessor processor(empty_particles);

    // Test that end_event works
    processor.end_event();
}

//---------------------------------------------------------------------------//

TEST_F(TrackProcessorTest, primary_registration)
{
    auto particles = make_particles();
    TrackProcessor processor(particles);

    // Create a primary track
    auto primary_track = std::make_unique<G4Track>(
        new G4DynamicParticle(particles[0], G4ThreeVector(1, 0, 0)),
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
    PrimaryId primary_id = processor.register_primary(*primary_track);

    // Verify primary ID
    EXPECT_EQ(0, primary_id.unchecked_get());

    // Verify user information was taken from the primary track
    EXPECT_EQ(nullptr, primary_track->GetUserInformation());

    // Test that process information can be retrieved by restoring the track
    G4Track& test_restored = processor.restore_track(ParticleId{0}, primary_id);
    EXPECT_EQ(mock_process.get(), test_restored.GetCreatorProcess());
    EXPECT_EQ(123, test_restored.GetTrackID());

    // Register another primary
    auto primary_track2 = std::make_unique<G4Track>(
        new G4DynamicParticle(particles[1], G4ThreeVector(0, 1, 0)),
        0.0,
        G4ThreeVector(1, 1, 1));
    primary_track2->SetTrackID(456);
    primary_track2->SetParentID(0);

    PrimaryId primary_id2 = processor.register_primary(*primary_track2);
    EXPECT_EQ(1, primary_id2.unchecked_get());
}

//---------------------------------------------------------------------------//

TEST_F(TrackProcessorTest, track_restoration)
{
    auto particles = make_particles();
    TrackProcessor processor(particles);

    // Create and register primary track with user information
    auto primary_track = std::make_unique<G4Track>(
        new G4DynamicParticle(particles[1], G4ThreeVector(0, 0, 1)),
        0.0,
        G4ThreeVector(0, 0, 0));
    primary_track->SetTrackID(789);
    primary_track->SetParentID(1);

    auto user_info = std::make_unique<MockUserTrackInformation>(99);
    primary_track->SetUserInformation(user_info.release());

    // Set creator process using mock process pointer
    auto mock_process = std::make_unique<MockProcess>("TestBremsstrahlung");
    primary_track->SetCreatorProcess(mock_process.get());

    PrimaryId primary_id = processor.register_primary(*primary_track);

    // Restore track for electron (particle ID 1) with primary information
    G4Track& restored_track
        = processor.restore_track(ParticleId{1}, primary_id);

    // Verify restored track properties
    EXPECT_EQ(789, restored_track.GetTrackID());
    EXPECT_EQ(1, restored_track.GetParentID());
    EXPECT_EQ(mock_process.get(), restored_track.GetCreatorProcess());
    EXPECT_EQ(&processor.step(), restored_track.GetStep());

    // Verify user information was restored
    auto* restored_user_info = dynamic_cast<MockUserTrackInformation*>(
        restored_track.GetUserInformation());
    ASSERT_NE(nullptr, restored_user_info);
    EXPECT_EQ(99, restored_user_info->value());

    // Verify particle type
    EXPECT_EQ(particles[1], restored_track.GetDefinition());
}

//---------------------------------------------------------------------------//

TEST_F(TrackProcessorTest, track_restoration_without_primary)
{
    auto particles = make_particles();
    TrackProcessor processor(particles);

    // Restore track without primary information (invalid PrimaryId)
    G4Track& restored_track
        = processor.restore_track(ParticleId{0}, PrimaryId{});

    // Verify basic track properties
    EXPECT_EQ(particles[0], restored_track.GetDefinition());
    EXPECT_EQ(0, restored_track.GetTrackID());
    EXPECT_EQ(0, restored_track.GetParentID());
    EXPECT_EQ(nullptr, restored_track.GetUserInformation());
    EXPECT_EQ(nullptr, restored_track.GetCreatorProcess());
}

//---------------------------------------------------------------------------//

TEST_F(TrackProcessorTest, end_event_cleanup)
{
    auto particles = make_particles();
    TrackProcessor processor(particles);

    // Register some primaries
    auto primary_track1 = std::make_unique<G4Track>(
        new G4DynamicParticle(particles[0], G4ThreeVector(1, 0, 0)),
        0.0,
        G4ThreeVector(0, 0, 0));
    primary_track1->SetTrackID(100);
    auto user_info1 = std::make_unique<MockUserTrackInformation>(10);
    primary_track1->SetUserInformation(user_info1.release());

    // Add different process pointers to test multiple process handling
    auto mock_process1 = std::make_unique<MockProcess>("TestProcess1");
    primary_track1->SetCreatorProcess(mock_process1.get());

    auto primary_track2 = std::make_unique<G4Track>(
        new G4DynamicParticle(particles[1], G4ThreeVector(0, 1, 0)),
        0.0,
        G4ThreeVector(0, 0, 0));
    primary_track2->SetTrackID(200);
    auto user_info2 = std::make_unique<MockUserTrackInformation>(20);
    primary_track2->SetUserInformation(user_info2.release());

    auto mock_process2 = std::make_unique<MockProcess>("TestProcess2");
    primary_track2->SetCreatorProcess(mock_process2.get());

    PrimaryId id1 = processor.register_primary(*primary_track1);
    PrimaryId id2 = processor.register_primary(*primary_track2);

    // Verify primaries are registered
    EXPECT_EQ(0, id1.unchecked_get());
    EXPECT_EQ(1, id2.unchecked_get());

    // Restore tracks to verify data exists
    G4Track& track1 = processor.restore_track(ParticleId{0}, id1);
    G4Track& track2 = processor.restore_track(ParticleId{1}, id2);
    EXPECT_EQ(100, track1.GetTrackID());
    EXPECT_EQ(200, track2.GetTrackID());

    // Verify that different process pointers are correctly restored
    EXPECT_EQ(mock_process1.get(), track1.GetCreatorProcess());
    EXPECT_EQ(mock_process2.get(), track2.GetCreatorProcess());
    EXPECT_NE(track1.GetCreatorProcess(), track2.GetCreatorProcess());

    // End event should clear reconstruction data
    processor.end_event();

    // Verify all tracks have cleared user information
    for (auto particle_id :
         range(ParticleId{static_cast<size_type>(particles.size())}))
    {
        G4Track& track = processor.restore_track(particle_id, PrimaryId{});
        EXPECT_EQ(nullptr, track.GetUserInformation());
    }
}

//---------------------------------------------------------------------------//

TEST_F(TrackProcessorTest, multiple_particle_types)
{
    auto particles = make_particles();
    TrackProcessor processor(particles);

    // Test all particle types can be restored
    for (auto i : range(particles.size()))
    {
        ParticleId particle_id{static_cast<size_type>(i)};
        G4Track& track = processor.restore_track(particle_id, PrimaryId{});

        EXPECT_EQ(particles[i], track.GetDefinition());
        EXPECT_EQ(0, track.GetTrackID());
        EXPECT_EQ(0, track.GetParentID());
    }
}

//---------------------------------------------------------------------------//

TEST_F(TrackProcessorTest, reconstruction_data_persistence)
{
    auto particles = make_particles();
    TrackProcessor processor(particles);

    // Create primary with complete information
    auto primary_track = std::make_unique<G4Track>(
        new G4DynamicParticle(particles[2], G4ThreeVector(1, 1, 1)),
        0.0,
        G4ThreeVector(10, 20, 30));
    primary_track->SetTrackID(999);
    primary_track->SetParentID(1);

    auto user_info = std::make_unique<MockUserTrackInformation>(777);
    primary_track->SetUserInformation(user_info.release());

    // Set creator process using mock process pointer
    auto mock_process = std::make_unique<MockProcess>("TestIonization");
    primary_track->SetCreatorProcess(mock_process.get());

    PrimaryId primary_id = processor.register_primary(*primary_track);

    // Test reconstruction data persists across multiple restore calls
    for (int i = 0; i < 3; ++i)
    {
        G4Track& restored = processor.restore_track(ParticleId{2}, primary_id);

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
}  // namespace test
}  // namespace detail
}  // namespace celeritas
