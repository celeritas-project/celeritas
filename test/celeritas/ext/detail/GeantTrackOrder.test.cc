//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantTrackOrder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/detail/GeantTrackOrder.hh"

#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#include <G4DynamicParticle.hh>
#include <G4Electron.hh>
#include <G4Gamma.hh>
#include <G4Positron.hh>
#include <G4Track.hh>

#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
struct TrackState
{
    G4ParticleDefinition const* particle{G4Electron::Definition()};
    double energy{1};
    double time{0};
    G4ThreeVector pos{0, 0, 0};
    G4ThreeVector dir{0, 0, 1};
    double weight{1};
    int track_id{1};
};

std::unique_ptr<G4Track> make_track(TrackState const& s)
{
    auto result = std::make_unique<G4Track>(
        new G4DynamicParticle(s.particle, s.dir, s.energy), s.time, s.pos);
    result->SetWeight(s.weight);
    result->SetTrackID(s.track_id);
    return result;
}

class GeantTrackOrderTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override
    {
        // Each state is ordered after the previous one, differing in the
        // listed key
        TrackState s;
        s.particle = G4Electron::Definition();  // PDG 11
        states_.push_back(s);
        s.track_id = 2;  // track ID
        states_.push_back(s);
        s.weight = 2;  // weight
        states_.push_back(s);
        s.dir = {0, 1, 0};  // direction y (with smaller z)
        states_.push_back(s);
        s.dir = {1, 0, 0};  // direction x
        states_.push_back(s);
        s.pos = {0, 0, 1};  // position z
        states_.push_back(s);
        s.pos = {0, 1, 0};  // position y (with smaller z)
        states_.push_back(s);
        s.pos = {1, 0, 0};  // position x
        states_.push_back(s);
        s.time = 1;  // global time
        states_.push_back(s);
        s.energy = 2;  // kinetic energy
        states_.push_back(s);
        s.particle = G4Gamma::Definition();  // PDG 22
        states_.push_back(s);

        // Positron (PDG -11) is first even with larger other keys
        TrackState first;
        first.particle = G4Positron::Definition();
        first.energy = 100;
        first.track_id = 100;
        states_.insert(states_.begin(), first);
    }

    std::vector<TrackState> states_;
};

TEST_F(GeantTrackOrderTest, strict_weak_order)
{
    GeantTrackOrder order;
    for (auto i : range(states_.size()))
    {
        auto lhs = make_track(states_[i]);
        EXPECT_FALSE(order(*lhs, *lhs)) << "irreflexive at " << i;
        for (auto j : range(i + 1, states_.size()))
        {
            auto rhs = make_track(states_[j]);
            EXPECT_TRUE(order(*lhs, *rhs)) << i << " < " << j;
            EXPECT_FALSE(order(*rhs, *lhs)) << j << " !< " << i;
        }
    }
}

TEST_F(GeantTrackOrderTest, independent_of_input_order)
{
    std::vector<std::unique_ptr<G4Track>> tracks;
    for (auto const& s : states_)
    {
        tracks.push_back(make_track(s));
    }
    std::vector<G4Track const*> expected;
    for (auto const& t : tracks)
    {
        expected.push_back(t.get());
    }

    auto compare = [](auto const& lhs, auto const& rhs) {
        return GeantTrackOrder{}(*lhs, *rhs);
    };
    std::mt19937 rng{12345u};
    for ([[maybe_unused]] auto i : range(10))
    {
        std::shuffle(tracks.begin(), tracks.end(), rng);
        std::sort(tracks.begin(), tracks.end(), compare);

        std::vector<G4Track const*> actual;
        for (auto const& t : tracks)
        {
            actual.push_back(t.get());
        }
        EXPECT_EQ(expected, actual);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
