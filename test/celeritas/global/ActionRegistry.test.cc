//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionRegistry.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/ActionRegistry.hh"

#include "celeritas/global/ActionRegistryOutput.hh"

#include "celeritas_test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Fake definition of CoreRef
template<MemSpace M>
struct CoreRef
{
};

namespace test
{
//---------------------------------------------------------------------------//

class MyExplicitAction final : public ExplicitActionInterface
{
  public:
    MyExplicitAction(ActionId ai, ActionOrder ao) : action_id_(ai), order_{ao}
    {
    }

    ActionId action_id() const final { return action_id_; }
    std::string label() const final { return "explicit"; }
    std::string description() const final { return "explicit action test"; }

    void execute(ParamsHostCRef const&, StateHostRef&) const final
    {
        ++host_count_;
    }
    void execute(ParamsDeviceCRef const&, StateDeviceRef&) const final
    {
        ++device_count_;
    }

    int host_count() const { return host_count_; }
    int device_count() const { return device_count_; }

    ActionOrder order() const final { return order_; }

  private:
    ActionId action_id_;
    ActionOrder order_;
    mutable int host_count_{0};
    mutable int device_count_{0};
};

class MyImplicitAction final : public ImplicitActionInterface,
                               public ConcreteAction
{
  public:
    // Construct with ID and label
    using ConcreteAction::ConcreteAction;
};

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ActionRegistryTest : public Test
{
  protected:
    void SetUp() override
    {
        EXPECT_EQ(ActionId{0}, mgr.next_id());
        EXPECT_EQ(0, mgr.num_actions());
        EXPECT_TRUE(mgr.empty());

        // Add actions
        auto impl1 = std::make_shared<MyImplicitAction>(mgr.next_id(), "impl1");
        mgr.insert(impl1);

        expl_action = std::make_shared<MyExplicitAction>(mgr.next_id(),
                                                         ActionOrder::pre);
        mgr.insert(expl_action);

        auto impl2 = std::make_shared<MyImplicitAction>(
            mgr.next_id(), "impl2", "the second implicit action");
        mgr.insert(impl2);
    }

    ActionRegistry mgr;
    std::shared_ptr<MyExplicitAction> expl_action;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ActionRegistryTest, accessors)
{
    EXPECT_FALSE(mgr.empty());
    EXPECT_EQ(3, mgr.num_actions());

    // Find IDs
    auto expl_id = mgr.find_action("explicit");
    EXPECT_EQ(1, expl_id.unchecked_get());
    EXPECT_EQ(0, mgr.find_action("impl1").unchecked_get());
    EXPECT_EQ(ActionId{}, mgr.find_action("nonexistent"));

    // Access an action
    EXPECT_EQ("explicit action test", mgr.action(expl_id)->description());
    EXPECT_EQ("explicit", mgr.id_to_label(expl_id));

    EXPECT_STREQ("pre", to_cstring(expl_action->order()));
}

TEST_F(ActionRegistryTest, output)
{
    // Create output handler from a shared pointer (with a null deleter)
    ActionRegistryOutput out(std::shared_ptr<ActionRegistry const>(
        &mgr, [](ActionRegistry const*) {}));
    EXPECT_EQ("actions", out.label());

    if (CELERITAS_USE_JSON)
    {
        EXPECT_EQ(
            R"json({"description":["","explicit action test","the second implicit action"],"label":["impl1","explicit","impl2"]})json",
            to_string(out))
            << "\n/*** REPLACE ***/\nR\"json(" << to_string(out)
            << ")json\"\n/******/";
    }
}

TEST_F(ActionRegistryTest, errors)
{
    // Incorrect ID
    EXPECT_THROW(
        mgr.insert(std::make_shared<MyImplicitAction>(ActionId{100}, "impl3")),
        RuntimeError);

    // Duplicate label
    EXPECT_THROW(
        mgr.insert(std::make_shared<MyImplicitAction>(mgr.next_id(), "impl1")),
        RuntimeError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
