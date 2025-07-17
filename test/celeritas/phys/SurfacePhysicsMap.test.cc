//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SurfacePhysicsMap.test.cc
//---------------------------------------------------------------------------//
#include "corecel/OpaqueIdUtils.hh"
#include "geocel/SurfaceParams.hh"
#include "geocel/SurfaceTestBase.hh"
#include "celeritas/phys/SurfaceModel.hh"
#include "celeritas/phys/SurfacePhysicsMapBuilder.hh"
#include "celeritas/phys/SurfacePhysicsMapData.hh"
#include "celeritas/phys/SurfacePhysicsMapView.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using S = SurfaceId;
using A = ActionId;

//---------------------------------------------------------------------------//
class MockSurfaceModel final : public SurfaceModel, public ConcreteAction
{
  public:
    MockSurfaceModel(ActionId id, VecSurfaceLayer surfaces)
        : ConcreteAction{id, std::to_string(id.get())}, surfaces_{surfaces}
    {
    }

    VecSurfaceLayer get_surfaces() const final { return surfaces_; }

  private:
    VecSurfaceLayer surfaces_;
};

//---------------------------------------------------------------------------//
// See geocel/SurfaceTestBase.hh for a description of surfaces: there are 9
class SurfacePhysicsMapTest : public SurfaceTestBase
{
  protected:
    void SetUp() override
    {
        SurfaceTestBase::SetUp();

        surfaces_ = SurfaceParams{this->make_many_surfaces_inp(), volumes_};
    }

    SurfaceParams surfaces_;
    HostVal<SurfacePhysicsMapData> host_;
};

TEST_F(SurfacePhysicsMapTest, typical)
{
    // Construct builder
    SurfacePhysicsMapBuilder add_surface_model(surfaces_, host_);

    // Add a model with some surfaces, which don't have to be ordered
    add_surface_model(MockSurfaceModel{A{0}, {S{3}, S{1}, S{4}, S{5}, S{7}}});
    add_surface_model(MockSurfaceModel{A{3}, {S{0}, S{2}}});
    add_surface_model(MockSurfaceModel{A{1}, {S{8}}});

    // Save reference to data
    NativeCRef<SurfacePhysicsMapData> ref;
    ref = host_;

    // Test surfaces
    std::vector<int> actions;
    std::vector<int> model_surfaces;
    for (auto sid : range(S{9}))
    {
        SurfacePhysicsMapView physics{ref, sid};
        auto action_id = physics.action_id();
        actions.push_back(id_to_int(action_id));
        if (action_id)
        {
            model_surfaces.push_back(id_to_int(physics.model_surface_id()));
        }
        else
        {
            // model_surface_id shouldn't be called if action wasn't assigned
            model_surfaces.push_back(-2);
        }
    }

    static int const expected_actions[] = {3, 0, 3, 0, 0, 0, -1, 0, 1};
    EXPECT_VEC_EQ(expected_actions, actions);
    static int const expected_model_surfaces[] = {0, 1, 1, 0, 2, 3, -2, 4, 0};
    EXPECT_VEC_EQ(expected_model_surfaces, model_surfaces);
}

TEST_F(SurfacePhysicsMapTest, errors)
{
    // Construct builder
    SurfacePhysicsMapBuilder add_surface_model(surfaces_, host_);

    // Empty model not allowed
    EXPECT_THROW(add_surface_model(MockSurfaceModel{A{0}, {}}), RuntimeError);
    // Duplicate action ID not allowed
    EXPECT_THROW(add_surface_model(MockSurfaceModel{A{0}, {S{1}}}),
                 RuntimeError);

    // Add a model
    add_surface_model(MockSurfaceModel{A{1}, {S{1}, S{3}}});
    // Multiple surfaces cannot have the same model
    EXPECT_THROW(add_surface_model(MockSurfaceModel{A{1}, {S{2}, S{3}}}),
                 RuntimeError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
