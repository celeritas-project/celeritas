//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalMockTestBase.cc
//---------------------------------------------------------------------------//
#include "OpticalMockTestBase.hh"

#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/Model.hh"
#include "celeritas/optical/ModelBuilder.hh"
#include "celeritas/optical/PhysicsParams.hh"
#include "celeritas/optical/detail/MfpBuilder.hh"

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

    constexpr real_type low_energy[] = {1e-6, 1e-3, 1};
    constexpr real_type high_energy[] = {1e3, 2e4, 3e5};

    model_builders.push_back(std::make_shared<MockModelBuilder>(
        "mock-optical-model-1",
        "mock-optical-description-1",
        std::vector<ImportPhysicsVector>{
            ImportPhysicsVector{
                linear, {low_energy[0], high_energy[0]}, {2.5, 3.7}},
            ImportPhysicsVector{
                linear, {low_energy[1], high_energy[1]}, {9.3, 12.3}},
            ImportPhysicsVector{
                linear, {low_energy[2], high_energy[2]}, {83.1, 128}}}));

    model_builders.push_back(std::make_shared<MockModelBuilder>(
        "mock-optical-model-2",
        "mock-optical-description-2",
        std::vector<ImportPhysicsVector>{
            ImportPhysicsVector{
                linear, {low_energy[1], high_energy[1]}, {8.1, 9.9}},
            ImportPhysicsVector{
                linear, {low_energy[2], high_energy[0]}, {2.7, 8.6}},
            ImportPhysicsVector{
                linear, {low_energy[0], high_energy[1]}, {91.1, 221}}}));

    model_builders.push_back(std::make_shared<MockModelBuilder>(
        "mock-optical-model-3",
        "mock-optical-description-3",
        std::vector<ImportPhysicsVector>{
            ImportPhysicsVector{
                linear, {low_energy[2], high_energy[1]}, {0.1, 4.9}},
            ImportPhysicsVector{
                linear, {low_energy[2], high_energy[2]}, {7.5, 18.2}},
            ImportPhysicsVector{
                linear, {low_energy[1], high_energy[2]}, {1.8, 19.8}}}));

    model_builders.push_back(std::make_shared<MockModelBuilder>(
        "mock-optical-model-4",
        "mock-optical-description-4",
        std::vector<ImportPhysicsVector>{
            ImportPhysicsVector{
                linear, {low_energy[0], high_energy[1]}, {1.2, 5.1}},
            ImportPhysicsVector{
                linear, {low_energy[1], high_energy[0]}, {11.1, 12.3}},
            ImportPhysicsVector{
                linear, {low_energy[0], high_energy[2]}, {180, 341}}}));

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

auto OpticalMockTestBase::build_optical_material() -> SPConstOpticalMaterial
{
    // Empty optical material data...
    MaterialParams::Input input;
    input.properties
        = std::vector<ImportOpticalProperty>(3, ImportOpticalProperty{});

    return std::make_shared<MaterialParams>(std::move(input));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
