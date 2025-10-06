//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/InvalidOrangeTestBase.cc
//---------------------------------------------------------------------------//
#include "InvalidOrangeTestBase.hh"

#include <memory>

#include "corecel/io/Join.hh"
#include "orange/OrangeInput.hh"
#include "orange/OrangeParams.hh"
#include "orange/orangeinp/CsgObject.hh"
#include "orange/orangeinp/InputBuilder.hh"
#include "orange/orangeinp/Shape.hh"
#include "orange/orangeinp/Transformed.hh"
#include "orange/orangeinp/UnitProto.hh"
#include "celeritas/Units.hh"
#include "celeritas/geo/GeoMaterialParams.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
using SPConstObject = std::shared_ptr<orangeinp::ObjectInterface const>;

SPConstObject make_sph(std::string&& label, real_type radius)
{
    using orangeinp::Sphere;
    using orangeinp::SphereShape;
    return std::make_shared<SphereShape>(std::move(label), Sphere{radius});
}

SPConstObject
make_sph(std::string&& label, real_type radius, Real3 const& trans)
{
    return std::make_shared<orangeinp::Transformed>(
        make_sph(std::move(label), radius), Translation{trans});
}

auto make_material(std::string&& label, GeoMatId::size_type m, SPConstObject obj)
{
    CELER_EXPECT(obj);
    orangeinp::UnitProto::MaterialInput result;
    result.interior = std::move(obj);
    result.fill = GeoMatId{m};
    result.label = std::move(label);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
auto InvalidOrangeTestBase::build_geometry() -> SPConstCoreGeo
{
    using namespace orangeinp;
    constexpr auto inside = Sense::inside;
    constexpr auto outside = Sense::outside;
    constexpr real_type cm{units::centimeter};

    // Construct shapes
    auto outer = make_sph("outer", 15 * cm);
    auto inner = make_sph("inner", 10 * cm);
    auto left = make_sph("left", 1 * cm, Real3{-5 * cm, 0, 0});
    auto center = make_sph("center", 1 * cm);
    auto right = make_sph("right", 1 * cm, Real3{5 * cm, 0, 0});

    // Construct proto (volumes, materials)
    UnitProto::Input inp;
    inp.boundary.interior = outer;
    inp.boundary.zorder = ZOrder::media;
    inp.label = "world";
    inp.materials.push_back(make_material("interior",
                                          0,
                                          make_rdv("interior-fill",
                                                   {{inside, inner},
                                                    {outside, left},
                                                    {outside, center},
                                                    {outside, right}})));
    inp.materials.push_back(make_material("also-interior", 0, center));
    inp.materials.push_back(make_material(
        "world",
        1,
        make_rdv("world-shell", {{inside, outer}, {outside, inner}})));
    inp.materials.push_back(make_material("[missing material]", 2, right));

    // Construct input
    OrangeInput orangeinp = InputBuilder{[&] {
        InputBuilder::Options opts;
        opts.tol = Tolerance<>::from_default(1 * cm);
        return opts;
    }()}(UnitProto{std::move(inp)});

    // Construct geometry
    auto result = std::make_shared<OrangeParams>(std::move(orangeinp));
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
    return result;
#else
    CELER_DISCARD(result);
    CELER_NOT_CONFIGURED("ORANGE as runtime geometry");
#endif
}

//---------------------------------------------------------------------------//
auto InvalidOrangeTestBase::build_geomaterial() -> SPConstGeoMaterial
{
    using Input = GeoMaterialParams::Input;
    Input input;
    input.geometry = this->geometry();
    input.materials = this->material();
    input.volume_to_mat = Input::MapLabelMat{
        {"interior", PhysMatId{0}},
        {"also-interior", PhysMatId{0}},
        {"world", PhysMatId{1}},
    };
    return std::make_shared<GeoMaterialParams>(std::move(input));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
