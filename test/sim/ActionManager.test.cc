//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file sim/ActionManager.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/ActionManager.hh"

#include "celeritas/global/ActionManagerOutput.hh"

#include "celeritas_test.hh"

using namespace celeritas;

namespace celeritas
{
//---------------------------------------------------------------------------//
// Fake definition of CoreRef
template<MemSpace M>
struct CoreRef
{
};
//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//

class MyExplicitAction final : public ExplicitActionInterface
{
  public:
    explicit MyExplicitAction(ActionId ai) : action_id_(ai) {}

    ActionId    action_id() const final { return action_id_; }
    std::string label() const final { return "explicit"; }
    std::string description() const final { return "explicit action test"; }

    void execute(CoreHostRef const&) const final { ++host_count_; }
    void execute(CoreDeviceRef const&) const final { ++device_count_; }

    int host_count() const { return host_count_; }
    int device_count() const { return device_count_; }

  private:
    ActionId    action_id_;
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

class ActionManagerTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        EXPECT_EQ(ActionId{0}, mgr.next_id());
        EXPECT_EQ(0, mgr.num_actions());

        // Add actions
        auto impl1 = std::make_shared<MyImplicitAction>(mgr.next_id(), "impl1");
        mgr.insert(impl1);

        expl_action = std::make_shared<MyExplicitAction>(mgr.next_id());
        mgr.insert(expl_action);

        auto impl2 = std::make_shared<MyImplicitAction>(
            mgr.next_id(), "impl2", "the second implicit action");
        mgr.insert(impl2);
    }

    ActionManager                     mgr;
    std::shared_ptr<MyExplicitAction> expl_action;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ActionManagerTest, accessors)
{
    EXPECT_EQ(3, mgr.num_actions());

    // Find IDs
    auto expl_id = mgr.find_action("explicit");
    EXPECT_EQ(1, expl_id.unchecked_get());
    EXPECT_EQ(0, mgr.find_action("impl1").unchecked_get());
    EXPECT_EQ(ActionId{}, mgr.find_action("nonexistent"));

    // Access an action
    EXPECT_EQ("explicit action test", mgr.action(expl_id).description());
    EXPECT_EQ("explicit", mgr.id_to_label(expl_id));
}

TEST_F(ActionManagerTest, output)
{
    // Create output handler from a shared pointer (with a null deleter)
    ActionManagerOutput out(std::shared_ptr<const ActionManager>(
        &mgr, [](const ActionManager*) {}));
    EXPECT_EQ("actions", out.label());

    if (CELERITAS_USE_JSON)
    {
        EXPECT_EQ(
            R"json([{"label":"impl1"},{"description":"explicit action test","label":"explicit"},{"description":"the second implicit action","label":"impl2"}])json",
            to_string(out));
    }
}

TEST_F(ActionManagerTest, invocation)
{
    EXPECT_EQ(0, expl_action->device_count());
    EXPECT_EQ(0, expl_action->host_count());

    CoreRef<MemSpace::host>   host_data;
    CoreRef<MemSpace::device> device_data;
    auto                      expl_id = mgr.find_action("explicit");
    mgr.invoke(expl_id, host_data);
    EXPECT_EQ(0, expl_action->device_count());
    EXPECT_EQ(1, expl_action->host_count());
    mgr.invoke(expl_id, device_data);
    EXPECT_EQ(1, expl_action->device_count());
}

TEST_F(ActionManagerTest, errors)
{
    // Incorrect ID
    EXPECT_THROW(
        mgr.insert(std::make_shared<MyImplicitAction>(ActionId{100}, "impl3")),
        celeritas::RuntimeError);

    // Duplicate label
    EXPECT_THROW(
        mgr.insert(std::make_shared<MyImplicitAction>(mgr.next_id(), "impl1")),
        celeritas::RuntimeError);

    // Invoke a nonexplicit kernel
    CoreRef<MemSpace::host> host_data;
    EXPECT_THROW(mgr.invoke(ActionId{0}, host_data), celeritas::RuntimeError);
}
