//----------------------------------*-C++-*----------------------------------//
// Copyright 2025 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalMockModels.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

#include "celeritas/optical/MfpBuilder.hh"
#include "celeritas/optical/Model.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Mock grids for 4 models and 5 optical materials.
 */
template<class Functor>
std::vector<std::vector<std::vector<real_type>>>
build_expected_grids(Functor const& f)
{
    ModelId::size_type num_models = 4;
    OpticalMaterialId::size_type num_materials = 5;

    std::vector<std::vector<std::vector<real_type>>> grids;
    grids.reserve(num_models);
    for (auto model : range(ModelId{num_models}))
    {
        std::vector<std::vector<real_type>> model_grids;
        model_grids.reserve(num_materials);
        for (auto mat : range(OpticalMaterialId{num_materials}))
        {
            size_type n = (model.get() + 1) * 10 + mat.get();
            std::vector<real_type> grid;
            grid.reserve(n + 1);
            for (size_type i : range(n + 1))
            {
                grid.push_back(f(i, n));
            }
            model_grids.push_back(std::move(grid));
        }
        grids.push_back(std::move(model_grids));
    }
    return grids;
}

//---------------------------------------------------------------------------//
/*!
 * Mock MFP grid energies for given material and model.
 */
Span<real_type const>
expected_mfp_energy_grid(OpticalMaterialId mat, ModelId model)
{
    static std::vector<std::vector<std::vector<real_type>>> grids;

    if (grids.empty())
    {
        grids = build_expected_grids([](size_type i, size_type n) {
            return real_type{15} * std::log(real_type(i) / n + 1);
        });
    }

    CELER_EXPECT(model < grids.size());
    CELER_EXPECT(mat < grids[model.get()].size());

    return make_span(grids[model.get()][mat.get()]);
}

//---------------------------------------------------------------------------//
/*!
 * Mock MFP grid values (the path lengths) for given material and model.
 */
Span<real_type const>
expected_mfp_value_grid(OpticalMaterialId mat, ModelId model)
{
    static std::vector<std::vector<std::vector<real_type>>> grids;

    if (grids.empty())
    {
        grids = build_expected_grids(
            [](size_type i, size_type) { return real_type(i * i); });
    }

    CELER_EXPECT(model < grids.size());
    CELER_EXPECT(mat < grids[model.get()].size());

    return make_span(grids[model.get()][mat.get()]);
}

//---------------------------------------------------------------------------//
/*!
 * Mock model that builds MFP grids from test data.
 */
class MockModel : public Model
{
  public:
    MockModel(ActionId id)
        : Model(id,
                "mock-" + std::to_string(id.get()),
                "mock-description-" + std::to_string(id.get()))
    {
    }

    void build_mfps(OpticalMaterialId mat, MfpBuilder& builder) const final
    {
        ModelId model_id{this->action_id().get() - 1};
        builder(expected_mfp_energy_grid(mat, model_id),
                expected_mfp_value_grid(mat, model_id));
    }

    void step(CoreParams const&, CoreStateHost&) const final {}
    void step(CoreParams const&, CoreStateDevice&) const final {}
};

//---------------------------------------------------------------------------//
/*!
 * Simple builder for mock models.
 */
struct MockModelBuilder
{
    std::shared_ptr<Model> operator()(ActionId id) const
    {
        return std::make_shared<MockModel>(id);
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
