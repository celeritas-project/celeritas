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
using MatGrid = std::vector<inp::Grid>;
using ModelMatGrid = std::vector<MatGrid>;

//---------------------------------------------------------------------------//
/*!
 * Mock grids for 4 models and 5 optical materials.
 */
template<class FX, class FY>
ModelMatGrid build_expected_grids(FX const& get_x, FY const& get_y)
{
    ModelId::size_type num_models = 4;
    OpticalMaterialId::size_type num_materials = 5;

    ModelMatGrid grids;
    grids.reserve(num_models);
    for (auto model : range(ModelId{num_models}))
    {
        MatGrid mat_grids;
        mat_grids.reserve(num_materials);
        for (auto mat : range(OpticalMaterialId{num_materials}))
        {
            size_type n = (model.get() + 1) * 10 + mat.get();
            inp::Grid grid;
            grid.x.reserve(n + 1);
            grid.y.reserve(n + 1);
            for (size_type i : range(n + 1))
            {
                grid.x.push_back(get_x(i, n));
                grid.y.push_back(get_y(i, n));
            }
            mat_grids.push_back(std::move(grid));
        }
        grids.push_back(std::move(mat_grids));
    }
    return grids;
}

//---------------------------------------------------------------------------//
/*!
 * Mock MFP grid for the given material and model.
 */
inp::Grid const& expected_mfp_grid(OpticalMaterialId mat, ModelId model)
{
    static ModelMatGrid grids;

    if (grids.empty())
    {
        grids = build_expected_grids(
            [](size_type i, size_type n) {
                return 15 * std::log(real_type(i) / n + 1);
            },
            [](size_type i, size_type) { return real_type(i * i); });
    }

    CELER_EXPECT(model < grids.size());
    CELER_EXPECT(mat < grids[model.get()].size());

    return grids[model.get()][mat.get()];
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

    void build_mfps(OpticalMaterialId mat, MfpBuilder& build) const final
    {
        ModelId model_id{this->action_id().get() - 1};
        build(expected_mfp_grid(mat, model_id));
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
