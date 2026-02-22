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

class SurfaceImporterTest : public ::celeritas::test::GeantTestBase
{
  public:
    std::string_view gdml_basename() const override
    {
        return "full-optical-surfaces";
    }

    GeantImportDataSelection build_import_data_selection() const override
    {
        auto result = GeantTestBase::build_import_data_selection();
        result.processes |= GeantImportDataSelection::optical;
        return result;
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
        else
        {
            std::cout << "  Expected surface " << phys_surface.unchecked_get()
                      << " missing\n";
        }
    }

    for (auto const& [phys_surface, _] : actual)
    {
        auto expected_iter = expected.find(phys_surface);
        if (expected_iter == expected.end())
        {
            std::cout << "  Unexpected surface "
                      << phys_surface.unchecked_get() << " found\n";
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

    std::cout << "Checking roughness polished\n";
    check_map(expected.roughness.polished, actual.roughness.polished);
    std::cout << "Checking roughness smear\n";
    check_map(expected.roughness.smear, actual.roughness.smear);
    std::cout << "Checking roughness gaussian\n";
    check_map(expected.roughness.gaussian, actual.roughness.gaussian);

    std::cout << "Checking reflectivity fresnel\n";
    check_map(expected.reflectivity.fresnel, actual.reflectivity.fresnel);
    std::cout << "Checking reflectivity grid\n";
    check_map(expected.reflectivity.grid, actual.reflectivity.grid);

    std::cout << "Checking interaction trivial\n";
    check_map(expected.interaction.trivial, actual.interaction.trivial);
    std::cout << "Checking interaction dielectric\n";
    check_map(expected.interaction.dielectric, actual.interaction.dielectric);
    std::cout << "Checking interaction only reflection\n";
    check_map(expected.interaction.only_reflection,
              actual.interaction.only_reflection);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(SurfaceImporterTest, full_optical_surfaces)
{
    using PSI = PhysSurfaceId;
    using namespace ::celeritas::inp;

    GridReflection refl{
        Grid{{1e-06, 1e-05}, {1, 1}}, Grid::from_constant(0), Grid{}};

    using Mode = optical::ReflectionMode;
    ReflectionForm unified_ground;
    unified_ground.reflection_grids[Mode::specular_spike]
        = {{1e-06, 1e-05}, {0.1, 0.3}};
    unified_ground.reflection_grids[Mode::specular_lobe]
        = {{1e-06, 1e-05}, {0.2, 0.2}};
    unified_ground.reflection_grids[Mode::backscatter]
        = {{1e-06, 1e-05}, {0.3, 0.1}};

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
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(dielectric, from_dielectric(from_spike()));
    }
    {
        ++surf;
        // GLISUR dielectric-dielectric ground
        MATERIALS({});
        ROUGHNESS(smear, SmearRoughness{0.1});
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(dielectric, from_dielectric(from_lobe()));
    }
    {
        ++surf;
        // GLISUR dielectric-metal polished
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(dielectric, from_metal(from_spike()));
    }
    {
        ++surf;
        // GLISUR dielectric-metal ground
        MATERIALS({});
        ROUGHNESS(smear, SmearRoughness{0.3});
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(dielectric, from_metal(from_lobe()));
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric polished
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(dielectric, from_dielectric(from_spike()));
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric ground
        MATERIALS({});
        ROUGHNESS(gaussian, GaussianRoughness{0.4});
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(dielectric, from_dielectric(unified_ground));
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric polished front painted
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(only_reflection, Mode::specular_spike);
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric ground front painted
        MATERIALS({});
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(only_reflection, Mode::diffuse_lobe);
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric polished back painted
        MATERIALS({OptMatId{2}});

        // material-gap surface
        ROUGHNESS(gaussian, GaussianRoughness{0.7});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_dielectric(unified_ground));

        ++surf;
        // gap-wrapping surface
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(only_reflection, Mode::specular_spike);
    }
    {
        ++surf;
        // UNIFIED dielectric-dielectric ground back painted
        MATERIALS({OptMatId{3}});

        // material-gap surface
        ROUGHNESS(gaussian, GaussianRoughness{0.8});
        REFLECTIVITY(grid, refl);
        INTERACTION(dielectric, from_dielectric(unified_ground));

        ++surf;
        // gap-wrapping surface
        ROUGHNESS(polished, NoRoughness{});
        REFLECTIVITY(fresnel, FresnelReflection{});
        INTERACTION(only_reflection, Mode::diffuse_lobe);
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
