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
using S = PhysSurfaceId;
using M = SurfaceModelId;

//---------------------------------------------------------------------------//
class MockSurfaceModel final : public SurfaceModel
{
  public:
    MockSurfaceModel(SurfaceModelId id,
                     std::string_view label,
                     VecSurfaceLayer surfaces)
        : SurfaceModel{id, label}, surfaces_{surfaces}
    {
    }

    VecSurfaceLayer get_surfaces() const final { return surfaces_; }

  private:
    VecSurfaceLayer surfaces_;
};

//---------------------------------------------------------------------------//
// See geocel/SurfaceTestBase.hh for a description of surfaces: there are 9
class SurfacePhysicsMapTest : public ManySurfacesTestBase
{
  protected:
    using MSM = MockSurfaceModel;

    HostVal<SurfacePhysicsMapData> host_;
};

TEST_F(SurfacePhysicsMapTest, typical)
{
    // Construct builder
    SurfacePhysicsMapBuilder add_model(this->surfaces().num_surfaces() + 1,
                                       host_);

    // Add a model with some surfaces, which don't have to be ordered
    add_model(MSM{M{0}, "A", {S{3}, S{1}, S{4}, S{5}, S{7}}});
    add_model(MSM{M{3}, "B", {S{0}, S{2}, S{9}}});
    add_model(MSM{M{1}, "C", {S{8}}});

    // Save reference to data
    NativeCRef<SurfacePhysicsMapData> ref;
    ref = host_;

    std::vector<int> actions;
    std::vector<int> model_surfaces;
    auto test_impl = [&](SurfacePhysicsMapView const& physics) {
        auto surface_model_id = physics.surface_model_id();
        actions.push_back(id_to_int(surface_model_id));
        if (surface_model_id)
        {
            model_surfaces.push_back(id_to_int(physics.internal_surface_id()));
        }
        else
        {
            // internal_surface_id shouldn't be called if action wasn't
            // assigned
            model_surfaces.push_back(-2);
        }
    };

    // Test surfaces
    for (auto sid : range(S{9}))
    {
        SurfacePhysicsMapView physics{ref, sid};
        test_impl(physics);
    }
    // Test default
    test_impl(SurfacePhysicsMapView{ref});

    static int const expected_actions[] = {3, 0, 3, 0, 0, 0, -1, 0, 1, 3};
    EXPECT_VEC_EQ(expected_actions, actions);
    static int const expected_model_surfaces[]
        = {0, 1, 1, 0, 2, 3, -2, 4, 0, 2};
    EXPECT_VEC_EQ(expected_model_surfaces, model_surfaces);
}

TEST_F(SurfacePhysicsMapTest, errors)
{
    // Construct builder
    SurfacePhysicsMapBuilder add_model(this->surfaces().num_surfaces(), host_);

    // Empty model not allowed
    EXPECT_THROW(add_model(MSM{M{0}, "A", {}}), RuntimeError);
    // Duplicate action ID not allowed
    EXPECT_THROW(add_model(MSM{M{0}, "B", {S{1}}}), RuntimeError);

    // Add a model
    add_model(MSM{M{1}, "C", {S{1}, S{3}}});
    // Multiple surfaces cannot have the same model
    EXPECT_THROW(add_model(MSM{M{1}, "D", {S{2}, S{3}}}), RuntimeError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
