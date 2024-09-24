//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalMockTestBase.cc
//---------------------------------------------------------------------------//
#include "OpticalMockTestBase.hh"

#include "celeritas/optical/PhysicsParams.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
class MockModel : public Model
{
  public:
    MockModel(ActionId id,
              std::string const& label,
              std::string const& desc,
              std::vector<ImportPhysicsVector> imported)
        : Model(id, label, desc), imported_(std::move(imported))
    {
    }

    void build_mfps(detail::MfpBuilder build) const override final
    {
        for (auto const& vec : imported_)
        {
            build(vec);
        }
    }

    void step(CoreParams const&, CoreStateHost&) const override final {}
    void step(CoreParams const&, CoreStateDevice&) const override final {}

  private:
    std::vector<ImportPhysicsVector> imported_;
};

class MockModelBuilder : public ModelBuilder
{
  public:
    MockModelBuilder(std::string const& label,
                     std::string const& desc,
                     std::vector<ImportPhysicsVector> imported)
        : label_(label), desc_(desc), imported_(std::move(imported))
    {
    }

    std::shared_ptr<Model> operator()(ActionId id) const override
    {
        return std::make_shared<MockModel>(id, label_, desc_, imported_);
    }

  private:
    std::string label_, desc_;
    std::vector<ImportPhysicsVector> imported_;
};

std::vector<std::shared_ptr<ModelBuilder const>> build_from_mocks()
{
    std::vector<std::shared_ptr<ModelBuilder const>> model_builders;

    constexpr auto linear = ImportPhysicsVectorType::linear;

    model_builders.push_back(
        std::make_shared<MockBuilder>("mock-optical-model-1",
                                      "mock-optical-description-1",
                                      {{linear, {0.01, 0.31}, {2.5, 3.7}},
                                       {linear, {0.12, 0.75}, {9.3, 12.3}},
                                       {linear, {0.001, 3.7}, {83.1, 128}}}));

    model_builders.push_back(
        std::make_shared<MockBuilder>("mock-optical-model-2",
                                      "mock-optical-description-2",
                                      {{linear, {0.07, 0.28}, {8.1, 9.9}},
                                       {linear, {0.55, 0.83}, {2.7, 8.6}},
                                       {linear, {0.012, 2.8}, {91.1, 221}}}));

    model_builders.push_back(
        std::make_shared<MockBuilder>("mock-optical-model-3",
                                      "mock-optical-description-3",
                                      {{linear, {2.7, 3.1}, {0.1, 4.9}},
                                       {linear, {3.8, 11.8}, {7.5, 18.2}},
                                       {linear, {7.9, 8.7}, {1.8, 19.8}}}));

    model_builders.push_back(
        std::make_shared<MockBuilder>("mock-optical-model-4",
                                      "mock-optical-description-4",
                                      {{linear, {9.9, 14.8}, {1.2, 5.1}},
                                       {linear, {4.8, 10.7}, {11.1, 12.3}},
                                       {linear, {0.11, 4.9}, {180, 341}}}));

    return model_builders;
}

auto OpticalMockTestBase::build_optical_physics() -> SPConstOpticalPhysics
{
    PhysicsParams::Input input;
    input.model_builders = build_from_mocks();
    input.action_registry = this->action_reg().get();
    input.options = PhysicsParamsOptions{};

    return std::make_shared<PhysicsParams>(std::move(input));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
