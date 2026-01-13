//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ActionRegistry.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/ActionRegistry.hh"

#include "corecel/sys/ActionRegistryOutput.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct MockParams
{
};

template<MemSpace M>
struct MockState
{
};

using MockBeginRunActionInterface
    = BeginRunActionInterface<MockParams, MockState>;
using MockStepActionInterface = StepActionInterface<MockParams, MockState>;

//---------------------------------------------------------------------------//

class MyStepAction final : public MockStepActionInterface,
                           public MockBeginRunActionInterface
{
  public:
    MyStepAction(ActionId ai, StepActionOrder ao) : action_id_(ai), order_{ao}
    {
    }

    ActionId action_id() const final { return action_id_; }
    std::string_view label() const final { return "step"; }
    std::string_view description() const final { return "step action test"; }

    void begin_run(CoreParams const&, CoreStateHost&) final
    {
        host_count_ = 0;
    }
    void begin_run(CoreParams const&, CoreStateDevice&) final
    {
        device_count_ = 0;
    }

    void step(CoreParams const&, CoreStateHost&) const final { ++host_count_; }
    void step(CoreParams const&, CoreStateDevice&) const final
    {
        ++device_count_;
    }

    int host_count() const { return host_count_; }
    int device_count() const { return device_count_; }

    StepActionOrder order() const final { return order_; }

  private:
    ActionId action_id_;
    StepActionOrder order_;
    mutable int host_count_{-100};
    mutable int device_count_{-100};
};

class MyImplicitAction final : public StaticConcreteAction
{
  public:
    // Construct with ID and label
    using StaticConcreteAction::StaticConcreteAction;
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

        step_action = std::make_shared<MyStepAction>(mgr.next_id(),
                                                     StepActionOrder::pre);
        mgr.insert(step_action);

        auto impl2 = std::make_shared<MyImplicitAction>(
            mgr.next_id(), "impl2", "the second implicit action");
        mgr.insert(impl2);
    }

    ActionRegistry mgr;
    std::shared_ptr<MyStepAction> step_action;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ActionRegistryTest, accessors)
{
    EXPECT_FALSE(mgr.empty());
    EXPECT_EQ(3, mgr.num_actions());

    // Find IDs
    auto step_id = mgr.find_action("step");
    EXPECT_EQ(1, step_id.unchecked_get());
    EXPECT_EQ(0, mgr.find_action("impl1").unchecked_get());
    EXPECT_EQ(ActionId{}, mgr.find_action("nonexistent"));

    // Access an action
    EXPECT_EQ("step action test", mgr.action(step_id)->description());
    EXPECT_EQ("step", mgr.id_to_label(step_id));

    EXPECT_STREQ("pre", to_cstring(step_action->order()));

    ASSERT_EQ(1, mgr.mutable_actions().size());
    EXPECT_EQ(step_action, mgr.mutable_actions().front());
}

TEST_F(ActionRegistryTest, output)
{
    // Create output handler from a shared pointer (with a null deleter)
    ActionRegistryOutput out(std::shared_ptr<ActionRegistry const>(
        &mgr, [](ActionRegistry const*) {}));
    EXPECT_EQ("actions", out.label());
    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"actions","description":["","step action test","the second implicit action"],"label":["impl1","step","impl2"]})json",
        to_string(out));
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
