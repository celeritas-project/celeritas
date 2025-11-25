//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxInterface.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/AuxInterface.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/data/AuxMockData.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxState.hh"
#include "corecel/data/AuxStateVec.hh"

#include "AuxMockParams.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class UserTest : public ::celeritas::test::Test
{
  protected:
    using VecInt = AuxMockParams::VecInt;

    void SetUp() override {}
};

TEST_F(UserTest, params)
{
    AuxParamsRegistry registry;
    AuxParamsRegistry const& creg = registry;
    EXPECT_EQ(0, registry.size());
    EXPECT_EQ(AuxId{0}, registry.next_id());

    // Add a params
    auto mock = std::make_shared<AuxMockParams>(
        "mock1", registry.next_id(), 123, VecInt{1, 2, 3, 4});
    registry.insert(mock);
    EXPECT_EQ(1, registry.size());
    EXPECT_EQ(mock.get(), registry.at(AuxId{0}).get());
    EXPECT_EQ(mock.get(), creg.at(AuxId{0}).get());
    EXPECT_EQ(AuxId{0}, registry.find("mock1"));
    EXPECT_EQ("mock1", registry.id_to_label(AuxId{0}));

    // Insertion (wrong ID) should be prohibited
    EXPECT_THROW(registry.insert(std::make_shared<AuxMockParams>(
                     "mock2", AuxId{2}, 123, VecInt{1, 2, 3, 4})),
                 RuntimeError);
    // Same name reinsertion should be prohibited
    EXPECT_THROW(registry.insert(std::make_shared<AuxMockParams>(
                     "mock1", registry.next_id(), 234, VecInt{1, 2, 3, 4})),
                 RuntimeError);

    // Add a second
    auto mock2 = std::make_shared<AuxMockParams>(
        "mock2", registry.next_id(), 234, VecInt{1, 2});
    registry.insert(mock2);
    EXPECT_EQ(2, registry.size());
    EXPECT_EQ(AuxId{1}, registry.find("mock2"));
}

TEST_F(UserTest, state_host)
{
    using StateT = AuxState<AuxMockStateData, MemSpace::host>;

    AuxParamsRegistry registry;
    auto mock = std::make_shared<AuxMockParams>(
        "mock1", registry.next_id(), 123, VecInt{1, 2, 3, 4});
    registry.insert(mock);
    auto mock2 = std::make_shared<AuxMockParams>(
        "mock2", registry.next_id(), 234, VecInt{1, 2});
    registry.insert(mock2);

    // Create a state vector (in practice, this is created and owned by the
    // CoreState)
    AuxStateVec states{registry, MemSpace::host, StreamId{1}, 128};
    EXPECT_EQ(2, states.size());

    {
        // Check the first state
        auto* sptr = dynamic_cast<StateT*>(&states.at(AuxId{0}));
        ASSERT_TRUE(sptr);
        EXPECT_TRUE(*sptr);
        EXPECT_EQ(128, sptr->size());

        HostRef<AuxMockStateData>& data = sptr->ref();
        EXPECT_EQ(StreamId{1}, data.stream);
        EXPECT_EQ(128, data.size());
        EXPECT_EQ(128, data.local_state.size());
        EXPECT_EQ(123, data.counts.size());

        // Use the params helper to get the device reference more easily
        // (it should be a pointer to the exact same data)
        auto& sref = mock->state_ref<MemSpace::host>(states);
        EXPECT_EQ(&data, &sref);
    }
    {
        // Check the second state
        auto* sptr = dynamic_cast<StateT*>(&states.at(AuxId{1}));

        ASSERT_TRUE(sptr);
        EXPECT_TRUE(*sptr);
        EXPECT_EQ(128, sptr->size());

        HostRef<AuxMockStateData>& data = sptr->ref();
        EXPECT_EQ(StreamId{1}, data.stream);
        EXPECT_EQ(128, data.local_state.size());
        EXPECT_EQ(234, data.counts.size());

        // Use the params helper to get the device reference more easily
        // (it should be a pointer to the exact same data)
        auto& sref = mock2->state_ref<MemSpace::host>(states);
        EXPECT_EQ(&data, &sref);
    }
    {
        // Check 'get'
        auto& s = get<StateT>(states, AuxId{1});
        EXPECT_EQ(128, s.size());
    }
    if (CELERITAS_DEBUG)
    {
        // Getting the wrong type/memspace should be a validation error (but
        // it's checked only when CELERITAS_DEBUG)
        EXPECT_THROW(mock2->state_ref<MemSpace::device>(states), RuntimeError);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
