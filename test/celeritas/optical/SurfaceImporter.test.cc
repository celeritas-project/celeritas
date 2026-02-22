//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfaceImporter.test.cc
//---------------------------------------------------------------------------//

#include <iostream>

#include "geocel/SurfaceParams.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/Physics.hh"
#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/io/ImportData.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class SurfaceImporterTestBase : public ::celeritas::test::GeantTestBase
{
  public:
    GeantImportDataSelection build_import_data_selection() const override
    {
        auto result = GeantTestBase::build_import_data_selection();
        result.processes |= GeantImportDataSelection::optical;
        return result;
    }
};

class SurfaceImporterOpticalSurfacesTest : public SurfaceImporterTestBase
{
  public:
    std::string_view gdml_basename() const override
    {
        return "optical-surfaces";
    }
};

class SurfaceImporterFullOpticalSurfacesTest : public SurfaceImporterTestBase
{
  public:
    std::string_view gdml_basename() const override
    {
        return "full-optical-surfaces";
    }
};

//---------------------------------------------------------------------------//

template<class T>
void check_input(T const& expected, T const& actual)
{
    EXPECT_EQ(expected, actual);
}

#define CHECK_DECL(TYPE) \
    template<>           \
    void check_input<TYPE>(TYPE const& expected, TYPE const& actual)

CHECK_DECL(inp::Grid)
{
    EXPECT_VEC_SOFT_EQ(expected.x, actual.x);
    EXPECT_VEC_SOFT_EQ(expected.y, actual.y);
}

CHECK_DECL(inp::NoRoughness)
{
    CELER_DISCARD(expected);
    CELER_DISCARD(actual);
}

CHECK_DECL(inp::SmearRoughness)
{
    EXPECT_SOFT_EQ(expected.roughness, actual.roughness);
}

CHECK_DECL(inp::GaussianRoughness)
{
    EXPECT_SOFT_EQ(expected.sigma_alpha, actual.sigma_alpha);
}

CHECK_DECL(inp::FresnelReflection)
{
    CELER_DISCARD(expected);
    CELER_DISCARD(actual);
}

CHECK_DECL(inp::GridReflection)
{
    check_input(expected.reflectivity, actual.reflectivity);
}

CHECK_DECL(inp::ReflectionForm)
{
    for (auto mode : range(optical::ReflectionMode::size_))
    {
        check_input(expected.reflection_grids[mode],
                    actual.reflection_grids[mode]);
    }
}

CHECK_DECL(inp::DielectricInteraction)
{
    EXPECT_EQ(expected.is_metal, actual.is_metal);
    check_input(expected.reflection, actual.reflection);
}

#undef CHECK_DECL

template<class T>
void check_map(std::map<PhysSurfaceId, T> const& expected,
               std::map<PhysSurfaceId, T> const& actual)
{
    EXPECT_EQ(expected.size(), actual.size());

    for (auto const& [phys_surface, expected_input] : expected)
    {
        auto actual_input = actual.find(phys_surface);
        EXPECT_TRUE(actual_input != actual.end());
        if (actual_input != actual.end())
        {
            check_input(expected_input, actual_input->second);
        }
    }
}

void check_input(inp::SurfacePhysics const& expected,
                 inp::SurfacePhysics const& actual)
{
    // Check number of geometric surfaces
    EXPECT_EQ(expected.materials.size(), actual.materials.size());

    // Compare interstitial materials
    for (auto surface_id : range(expected.materials.size()))
    {
        EXPECT_VEC_EQ(expected.materials[surface_id],
                      actual.materials[surface_id]);
    }

    check_map(expected.roughness.polished, actual.roughness.polished);
    check_map(expected.roughness.smear, actual.roughness.smear);
    check_map(expected.roughness.gaussian, actual.roughness.gaussian);

    check_map(expected.reflectivity.fresnel, actual.reflectivity.fresnel);
    check_map(expected.reflectivity.grid, actual.reflectivity.grid);

    check_map(expected.interaction.trivial, actual.interaction.trivial);
    check_map(expected.interaction.dielectric, actual.interaction.dielectric);
    check_map(expected.interaction.only_reflection,
              actual.interaction.only_reflection);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// TEST_F(SurfaceImporterOpticalSurfacesTest, optical_surfaces)
// {
//     using PSI = PhysSurfaceId;
//     using namespace ::celeritas::inp;
//
//     SurfacePhysics expected_input;
//
//     expected_input.materials = {
//         {},
//         {},
//         {},
//         {},
//         {},
//         {},
//     };
//
//     expected_input.roughness.polished = {
//         {PSI{0}, NoRoughness{}},
//         {PSI{3}, NoRoughness{}},
//         {PSI{5}, NoRoughness{}},
//     };
//
//     expected_input.roughness.smear = {
//         {PSI{1}, SmearRoughness{0.1}},
//         {PSI{4}, SmearRoughness{0.3}},
//     };
//
//     expected_input.roughness.gaussian = {
//         {PSI{2}, inp::GaussianRoughness{1}},
//     };
//
//     {
//         std::vector<double> grid_refl_x{1.65e-6, 1e-05};
//         expected_input.reflectivity.grid = {
//             {PSI{0}, GridReflection{grid_refl_x, {0.5, 0.5}}},
//             {PSI{1}, GridReflection{grid_refl_x, {0.6, 0.6}}},
//             {PSI{2}, GridReflection{grid_refl_x, {0.7, 0.7}}},
//             {PSI{3}, GridReflection{grid_refl_x, {0.8, 0.8}}},
//             {PSI{4}, GridReflection{grid_refl_x, {0.9, 0.9}}},
//         };
//     }
//
//     expected_input.reflectivity.fresnel = {
//         {PSI{5}, FresnelReflection{}},
//     };
//
//     expected_input.interaction.dielectric = {
//         {PSI{0},
//          DielectricInteraction{inp::ReflectionForm::from_spike(), false}},
//         {PSI{1}, DielectricInteraction{inp::ReflectionForm::from_lobe(),
//         false}}, {PSI{2},
//         DielectricInteraction{inp::ReflectionForm::from_lobe(), false}},
//         {PSI{3}, DielectricInteraction{inp::ReflectionForm::from_spike(),
//         true}}, {PSI{4},
//         DielectricInteraction{inp::ReflectionForm::from_lobe(), true}},
//         {PSI{5},
//          DielectricInteraction{inp::ReflectionForm::from_spike(), false}},
//     };
//     {
//         std::vector<double> grid_sc_x{2e-06, 8e-06};
//
//         auto& g = expected_input.interaction.dielectric[PSI(2)]
//                       .reflection.reflection_grids;
//
//         using Mode = optical::ReflectionMode;
//         g[Mode::specular_spike] = {grid_sc_x, {0.1, 0.3}};
//         g[Mode::specular_lobe] = {grid_sc_x, {0.2, 0.2}};
//         g[Mode::backscatter] = {grid_sc_x, {0.3, 0.1}};
//     }
//
//     expected_input.interaction.trivial = {};
//
//     check_input(expected_input,
//     this->imported_data().optical_physics.surfaces);
// }

//---------------------------------------------------------------------------//
TEST_F(SurfaceImporterFullOpticalSurfacesTest, full_optical_surfaces)
{
    using PSI = PhysSurfaceId;
    using namespace ::celeritas::inp;

    GridReflection refl{Grid{{1e-06, 1e-05}, {1, 1}}};

    using Mode = optical::ReflectionMode;
    ReflectionForm unified_ground;
    unified_ground.reflection_grids[Mode::specular_spike]
        = {{1e-06, 1e-05}, {0.1, 0.3}};
    unified_ground.reflection_grids[Mode::specular_lobe]
        = {{1e-06, 1e-05}, {0.2, 0.2}};
    unified_ground.reflection_grids[Mode::backscatter]
        = {{1e-06, 1e-05}, {0.3, 0.1}};

    OptMatId air_id{2};

    SurfacePhysics expected_input;

#define MATERIALS(VALUE) expected_input.materials.push_back(VALUE)
#define ROUGHNESS(TYPE, VALUE) \
    expected_input.roughness.TYPE.emplace(surf, VALUE)
#define REFLECTIVITY(TYPE, VALUE) \
    expected_input.reflectivity.TYPE.emplace(surf, VALUE)
#define INTERACTION(TYPE, VALUE) \
    expected_input.interaction.TYPE.emplace(surf, VALUE)

    auto from_dielectric = DielectricInteraction::from_dielectric;
    auto from_metal = DielectricInteraction::from_metal;
    auto from_spike = ReflectionForm::from_spike;
    auto from_lobe = ReflectionForm::from_lobe;

    PSI surf{0};
    {
        // GLISUR dielectric-dielectric polished
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_dielectric(from_spike()));
    }
    {
        ++surf;
        // GLISUR dielectric-dielectric ground
        MATERIALS({});
        ROUGHNESS(smear, SmearRoughness{0.1});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_dielectric(from_lobe()));
    }
    {
        ++surf;
        // GLISUR dielectric-metal polished
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_metal(from_spike()));
    }
    {
        ++surf;
        // GLISUR dielectric-metal ground
        MATERIALS({});
        ROUGHNESS(smear, SmearRoughness{0.3});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_metal(from_lobe()));
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric polished
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_dielectric(from_spike()));
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric ground
        MATERIALS({});
        ROUGHNESS(gaussian, GaussianRoughness{0.4});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_dielectric(unified_ground));
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric polished front painted
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(grid, refl);
        INTERACTION(only_reflection, Mode::specular_spike);
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric ground front painted
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(grid, refl);
        INTERACTION(only_reflection, Mode::diffuse_lobe);
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric polished back painted
        MATERIALS({air_id});

        // material-gap surface
        ROUGHNESS(gaussian, GaussianRoughness{0.7});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_dielectric(unified_ground));

        // gap-wrapping surface
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(grid, refl);
        INTERACTION(only_reflection, Mode::specular_spike);
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric ground back painted
        MATERIALS({air_id});

        // material-gap surface
        ROUGHNESS(gaussian, GaussianRoughness{0.8});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_dielectric(unified_ground));

        // gap-wrapping surface
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(grid, refl);
        INTERACTION(only_reflection, Mode::specular_lobe);
    }
    {
        ++surf;
        // UNIFIED dielectric-metal polished
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_metal(from_spike()));
    }
    {
        ++surf;
        // UNIFIED dielectric-metal ground
        MATERIALS({});
        ROUGHNESS(gaussian, GaussianRoughness{1.0});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_metal(unified_ground));
    }
    {
        ++surf;
        // Default Surface
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(dielectric, from_dielectric(from_spike()));
    }

#undef MATERIALS
#undef ROUGHNESS
#undef REFLECTIVITY
#undef INTERACTION

    check_input(expected_input, this->imported_data().optical_physics.surfaces);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
