//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/User.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/UserInterface.hh"
#include "corecel/data/UserParamsRegistry.hh"
#include "corecel/data/UserStateVec.hh"

#include "UserMockParams.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class UserTest : public ::celeritas::test::Test
{
  protected:
    using VecInt = UserMockParams::VecInt;

    void SetUp() override {}
};

TEST_F(UserTest, params)
{
    UserParamsRegistry registry;
    UserParamsRegistry const& creg = registry;
    EXPECT_EQ(0, registry.size());
    EXPECT_EQ(UserId{0}, registry.next_id());

    // Add a params
    auto mock = std::make_shared<UserMockParams>(
        "mock1", registry.next_id(), 123, VecInt{1, 2, 3, 4});
    registry.insert(mock);
    EXPECT_EQ(1, registry.size());
    EXPECT_EQ(mock.get(), registry.at(UserId{0}).get());
    EXPECT_EQ(mock.get(), creg.at(UserId{0}).get());
    EXPECT_EQ(UserId{0}, registry.find("mock1"));
    EXPECT_EQ("mock1", registry.id_to_label(UserId{0}));

    // Insertion (wrong ID) should be prohibited
    EXPECT_THROW(registry.insert(std::make_shared<UserMockParams>(
                     "mock2", UserId{2}, 123, VecInt{1, 2, 3, 4})),
                 RuntimeError);
    // Same name reinsertion should be prohibited
    EXPECT_THROW(registry.insert(std::make_shared<UserMockParams>(
                     "mock1", registry.next_id(), 234, VecInt{1, 2, 3, 4})),
                 RuntimeError);

    // Add a second
    auto mock2 = std::make_shared<UserMockParams>(
        "mock2", registry.next_id(), 234, VecInt{1, 2});
    registry.insert(mock2);
    EXPECT_EQ(2, registry.size());
    EXPECT_EQ(UserId{1}, registry.find("mock2"));
}

TEST_F(UserTest, state_host)
{
    using StateT = UserMockParams::StateT<MemSpace::host>;

    UserParamsRegistry registry;
    auto mock = std::make_shared<UserMockParams>(
        "mock1", registry.next_id(), 123, VecInt{1, 2, 3, 4});
    registry.insert(mock);
    auto mock2 = std::make_shared<UserMockParams>(
        "mock2", registry.next_id(), 234, VecInt{1, 2});
    registry.insert(mock2);

    // Create a state vector
    UserStateVec states{registry, MemSpace::host, StreamId{1}, 128};
    EXPECT_EQ(2, states.size());

    {
        // Check the first state
        auto* svec = dynamic_cast<StateT*>(&states.at(UserId{0}));
        ASSERT_TRUE(svec);
        EXPECT_TRUE(*svec);
        EXPECT_EQ(128, svec->size());

        HostRef<UserMockStateData>& data = svec->ref();
        EXPECT_EQ(StreamId{1}, data.stream);
        EXPECT_EQ(128, data.size());
        EXPECT_EQ(128, data.local_state.size());
        EXPECT_EQ(123, data.counts.size());
    }
    {
        // Check the second state
        auto* svec = dynamic_cast<StateT*>(&states.at(UserId{1}));
        ASSERT_TRUE(svec);
        EXPECT_TRUE(*svec);
        EXPECT_EQ(128, svec->size());

        HostRef<UserMockStateData>& data = svec->ref();
        EXPECT_EQ(StreamId{1}, data.stream);
        EXPECT_EQ(128, data.local_state.size());
        EXPECT_EQ(234, data.counts.size());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
