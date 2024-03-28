//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/SolidConverter.cc
//---------------------------------------------------------------------------//
#include "SolidConverter.hh"

#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include <G4BooleanSolid.hh>
#include <G4Box.hh>
#include <G4Cons.hh>
#include <G4CutTubs.hh>
#include <G4DisplacedSolid.hh>
#include <G4Ellipsoid.hh>
#include <G4EllipticalCone.hh>
#include <G4EllipticalTube.hh>
#include <G4ExtrudedSolid.hh>
#include <G4GenericPolycone.hh>
#include <G4GenericTrap.hh>
#include <G4Hype.hh>
#include <G4IntersectionSolid.hh>
#include <G4Orb.hh>
#include <G4Para.hh>
#include <G4Paraboloid.hh>
#include <G4Polycone.hh>
#include <G4Polyhedra.hh>
#include <G4ReflectedSolid.hh>
#include <G4RotationMatrix.hh>
#include <G4Sphere.hh>
#include <G4SubtractionSolid.hh>
#include <G4TessellatedSolid.hh>
#include <G4Tet.hh>
#include <G4ThreeVector.hh>
#include <G4Torus.hh>
#include <G4Trap.hh>
#include <G4Trd.hh>
#include <G4Tubs.hh>
#include <G4UnionSolid.hh>
#include <G4VSolid.hh>
#include <G4Version.hh>

#include "corecel/Constants.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "orange/orangeinp/CsgObject.hh"
#include "orange/orangeinp/Shape.hh"
#include "orange/orangeinp/Solid.hh"
#include "orange/orangeinp/Transformed.hh"

#include "Scaler.hh"
#include "Transformer.hh"

using namespace celeritas::orangeinp;

namespace celeritas
{
namespace g4org
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Return theta, phi angles for a G4Para or G4Trap given their symmetry axis.
 */
[[maybe_unused]] auto calculate_theta_phi(G4ThreeVector const& axis)
    -> std::pair<double, double>
{
    // The components of the symmetry axis for G4Para/Trap are alway encoded
    // as a vector (A.tan(theta)cos(phi), A.tan(theta)sin(phi), A).
    double const tan_theta_cos_phi = axis.x() / axis.z();
    double const tan_theta_sin_phi = axis.y() / axis.z();

    // Calculation taken from GetTheta/Phi implementations in Geant4 11
    double const theta
        = std::atan(std::sqrt(tan_theta_cos_phi * tan_theta_cos_phi
                              + tan_theta_sin_phi * tan_theta_sin_phi));
    double const phi = std::atan2(tan_theta_sin_phi, tan_theta_cos_phi);
    return {theta, phi};
}

//---------------------------------------------------------------------------//
template<class CR, class... Args>
auto make_shape(G4VSolid const& solid, Args&&... args)
{
    return std::make_shared<Shape<CR>>(std::string{solid.GetName()},
                                       CR{std::forward<Args>(args)...});
}

//---------------------------------------------------------------------------//
/*!
 * Get the enclosed azimuthal angle by a solid.
 *
 * This converts from radians (native, Geant4) to Turns.
 */
template<class S>
SolidEnclosedAngle get_azimuthal_wedge(S const& solid)
{
    return SolidEnclosedAngle{native_value_to<Turn>(solid.GetStartPhiAngle()),
                              native_value_to<Turn>(solid.GetDeltaPhiAngle())};
}

//---------------------------------------------------------------------------//
/*!
 * Get the enclosed polar angle by a solid.
 *
 * This converts from radians (native, Geant4) to Turns.
 */
template<class S>
SolidEnclosedAngle get_polar_wedge(S const& solid)
{
    return SolidEnclosedAngle{
        native_value_to<Turn>(solid.GetStartThetaAngle()),
        native_value_to<Turn>(solid.GetDeltaThetaAngle())};
}

//---------------------------------------------------------------------------//
template<class CR>
auto make_solid(G4VSolid const& solid,
                CR&& interior,
                std::optional<CR>&& excluded,
                SolidEnclosedAngle&& enclosed)
{
    return Solid<CR>::or_shape(std::string{solid.GetName()},
                               std::move(interior),
                               std::move(excluded),
                               std::move(enclosed));
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 solid to a CSG object.
 */
auto SolidConverter::operator()(arg_type solid_base) -> result_type
{
    auto [cache_iter, inserted] = cache_.insert({&solid_base, nullptr});
    if (inserted)
    {
        // First time converting the solid
        cache_iter->second = this->convert_impl(solid_base);
    }

    CELER_ENSURE(cache_iter->second);
    return cache_iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a geant4 solid to a VecGeom sphere with equivalent capacity.
 */
auto SolidConverter::to_sphere(arg_type solid_base) const -> result_type
{
    double vol = this->calc_capacity(solid_base);
    double radius = std::cbrt(vol / (4.0 / 3.0 * constants::pi));
    return make_shape<Sphere>(solid_base, radius);
}

//---------------------------------------------------------------------------//
/*!
 * Convert a solid that's not in the cache.
 */
auto SolidConverter::convert_impl(arg_type solid_base) -> result_type
{
    using ConvertFuncPtr = result_type (SolidConverter::*)(arg_type);
    using MapTypeConverter
        = std::unordered_map<std::type_index, ConvertFuncPtr>;

    // clang-format off
    #define SC_TYPE_FUNC(MIXED, LOWER) \
    {std::type_index(typeid(G4##MIXED)), &SolidConverter::LOWER}
    static const MapTypeConverter type_to_converter = {
        SC_TYPE_FUNC(Box              , box),
        SC_TYPE_FUNC(Cons             , cons),
        SC_TYPE_FUNC(CutTubs          , cuttubs),
        SC_TYPE_FUNC(DisplacedSolid   , displaced),
        SC_TYPE_FUNC(Ellipsoid        , ellipsoid),
        SC_TYPE_FUNC(EllipticalCone   , ellipticalcone),
        SC_TYPE_FUNC(EllipticalTube   , ellipticaltube),
        SC_TYPE_FUNC(ExtrudedSolid    , extrudedsolid),
        SC_TYPE_FUNC(GenericPolycone  , genericpolycone),
        SC_TYPE_FUNC(GenericTrap      , generictrap),
        SC_TYPE_FUNC(Hype             , hype),
        SC_TYPE_FUNC(IntersectionSolid, intersectionsolid),
        SC_TYPE_FUNC(Orb              , orb),
        SC_TYPE_FUNC(Para             , para),
        SC_TYPE_FUNC(Paraboloid       , paraboloid),
        SC_TYPE_FUNC(Polycone         , polycone),
        SC_TYPE_FUNC(Polyhedra        , polyhedra),
        SC_TYPE_FUNC(ReflectedSolid   , reflectedsolid),
        SC_TYPE_FUNC(Sphere           , sphere),
        SC_TYPE_FUNC(SubtractionSolid , subtractionsolid),
        SC_TYPE_FUNC(TessellatedSolid , tessellatedsolid),
        SC_TYPE_FUNC(Tet              , tet),
        SC_TYPE_FUNC(Torus            , torus),
        SC_TYPE_FUNC(Trap             , trap),
        SC_TYPE_FUNC(Trd              , trd),
        SC_TYPE_FUNC(Tubs             , tubs),
        SC_TYPE_FUNC(UnionSolid       , unionsolid),
    };
    // clang-format on
#undef SC_TYPE_FUNC

    // Look up converter function based on the solid's C++ type
    auto func_iter
        = type_to_converter.find(std::type_index(typeid(solid_base)));

    result_type result = nullptr;
    CELER_VALIDATE(func_iter != type_to_converter.end(),
                   << "unsupported solid type "
                   << TypeDemangler<G4VSolid>{}(solid_base));

    // Call our corresponding member function to convert the solid
    ConvertFuncPtr fp = func_iter->second;
    result = (this->*fp)(solid_base);
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
// CONVERTERS
//---------------------------------------------------------------------------//
//! Convert a box
auto SolidConverter::box(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Box const&>(solid_base);
    return make_shape<Box>(solid,
                           scale_.to<Real3>(solid.GetXHalfLength(),
                                            solid.GetYHalfLength(),
                                            solid.GetZHalfLength()));
}

//---------------------------------------------------------------------------//
//! Convert a cone section
auto SolidConverter::cons(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Cons const&>(solid_base);

    auto const outer_r = scale_.to<Cone::Real2>(solid.GetOuterRadiusMinusZ(),
                                                solid.GetOuterRadiusPlusZ());
    auto const inner_r = scale_.to<Cone::Real2>(solid.GetInnerRadiusMinusZ(),
                                                solid.GetInnerRadiusPlusZ());
    auto hh = scale_(solid.GetZHalfLength());

    if (outer_r[0] == outer_r[1])
    {
        std::optional<Cylinder> inner;
        if (inner_r[0] || inner_r[1])
        {
            inner = Cylinder{inner_r[0], hh};
        }

        return make_solid(solid,
                          Cylinder{outer_r[0], hh},
                          std::move(inner),
                          get_azimuthal_wedge(solid));
    }

    std::optional<Cone> inner;
    if (inner_r[0] || inner_r[1])
    {
        inner = Cone{inner_r, hh};
    }

    return make_solid(
        solid, Cone{outer_r, hh}, std::move(inner), get_azimuthal_wedge(solid));
}

//---------------------------------------------------------------------------//
//! Convert a cut tube
auto SolidConverter::cuttubs(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4CutTubs const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("cuttubs");
}

//---------------------------------------------------------------------------//
//! Convert a displaced solid
auto SolidConverter::displaced(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4DisplacedSolid const&>(solid_base);
    G4VSolid* g4daughter = solid.GetConstituentMovedSolid();
    CELER_ASSERT(g4daughter);
    auto daughter = (*this)(*g4daughter);
    return std::make_shared<Transformed>(
        daughter, transform_(solid.GetDirectTransform()));
}

//---------------------------------------------------------------------------//
//! Convert an ellipsoid
auto SolidConverter::ellipsoid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Ellipsoid const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("ellipsoid");
}

//---------------------------------------------------------------------------//
//! Convert an elliptical cone
auto SolidConverter::ellipticalcone(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4EllipticalCone const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("ellipticalcone");
}

//---------------------------------------------------------------------------//
//! Convert an elliptical tube
auto SolidConverter::ellipticaltube(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4EllipticalTube const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("ellipticaltube");
}

//---------------------------------------------------------------------------//
//! Convert an extruded solid
auto SolidConverter::extrudedsolid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4ExtrudedSolid const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("extrudedsolid");
}

//---------------------------------------------------------------------------//
//! Convert a generic polycone
auto SolidConverter::genericpolycone(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4GenericPolycone const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("genericpolycone");
}

//---------------------------------------------------------------------------//
//! Convert a generic trapezoid
auto SolidConverter::generictrap(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4GenericTrap const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("generictrap");
}

//---------------------------------------------------------------------------//
//! Convert a hyperbola
auto SolidConverter::hype(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Hype const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("hype");
}

//---------------------------------------------------------------------------//
//! Convert an intersection solid
auto SolidConverter::intersectionsolid(arg_type solid_base) -> result_type
{
    auto vols = this->make_bool_solids(
        dynamic_cast<G4BooleanSolid const&>(solid_base));
    return std::make_shared<AllObjects>(
        std::string{solid_base.GetName()},
        std::vector{std::move(vols[0]), std::move(vols[1])});
}

//---------------------------------------------------------------------------//
//! Convert an orb
auto SolidConverter::orb(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Orb const&>(solid_base);
    return make_shape<Sphere>(solid, scale_(solid.GetRadius()));
}

//---------------------------------------------------------------------------//
//! Convert a parallelepiped
auto SolidConverter::para(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Para const&>(solid_base);
#if G4VERSION_NUMBER >= 1100
    double const theta = solid.GetTheta();
    double const phi = solid.GetPhi();
#else
    CELER_NOT_IMPLEMENTED("older geant4, don't duplicate code");
#endif
    return make_shape<Parallelepiped>(
        solid,
        scale_.to<Real3>(solid.GetXHalfLength(),
                         solid.GetYHalfLength(),
                         solid.GetZHalfLength()),
        native_value_to<Turn>(std::atan(solid.GetTanAlpha())),
        native_value_to<Turn>(theta),
        native_value_to<Turn>(phi));
}

//---------------------------------------------------------------------------//
//! Convert a paraboloid
auto SolidConverter::paraboloid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Paraboloid const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("paraboloid");
}

//---------------------------------------------------------------------------//
//! Convert a polycone
auto SolidConverter::polycone(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Polycone const&>(solid_base);
    auto const& params = *solid.GetOriginalParameters();

    std::vector<double> zs(params.Num_z_planes);
    std::vector<double> rmin(zs.size());
    std::vector<double> rmax(zs.size());
    for (auto i : range(zs.size()))
    {
        zs[i] = scale_(params.Z_values[i]);
        rmin[i] = scale_(params.Rmin[i]);
        rmax[i] = scale_(params.Rmax[i]);
    }

    if (zs.size() == 2 && rmin[0] == 0 && rmin[1] == 0)
    {
        // Special case: displaced cone/cylinder
        double const hh = (zs[1] - zs[0]) / 2;
        result_type result;
        if (rmax[0] == rmax[1])
        {
            // Cylinder is a special case
            result = make_shape<Cylinder>(solid, rmax[0], hh);
        }
        else
        {
            result = make_shape<Cone>(solid, Cone::Real2{rmax[0], rmin[1]}, hh);
        }

        double dz = (zs[1] + zs[0]) / 2;
        if (dz != 0)
        {
            result = std::make_shared<Transformed>(std::move(result),
                                                   Translation{{0, 0, dz}});
        }
        return result;
    }

    CELER_NOT_IMPLEMENTED("polycone");
}

//---------------------------------------------------------------------------//
//! Convert a polyhedron
auto SolidConverter::polyhedra(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Polyhedra const&>(solid_base);
    auto const& params = *solid.GetOriginalParameters();

    // Opening angle: end - start phi
    double const radius_factor
        = std::cos(0.5 * params.Opening_angle / params.numSide);

    std::vector<double> zs(params.Num_z_planes);
    std::vector<double> rmin(zs.size());
    std::vector<double> rmax(zs.size());
    for (auto i : range(zs.size()))
    {
        zs[i] = scale_(params.Z_values[i]);
        rmin[i] = scale_(params.Rmin[i]) * radius_factor;
        rmax[i] = scale_(params.Rmax[i]) * radius_factor;
    }

    auto startphi = native_value_to<Turn>(solid.GetStartPhi());
    SolidEnclosedAngle angle(
        startphi, native_value_to<Turn>(solid.GetEndPhi()) - startphi);

    if (zs.size() == 2 && rmin[0] == rmin[1] && rmax[0] == rmax[1])
    {
        // A solid prism
        double const hh = (zs[1] - zs[0]) / 2;
        double const orientation = startphi.value() / params.numSide;

        if (rmin[0] != 0.0 || angle)
        {
            CELER_NOT_IMPLEMENTED("prism solid");
        }

        result_type result = make_shape<Prism>(
            solid, params.numSide, rmax[0], hh, orientation);

        double dz = (zs[1] + zs[0]) / 2;
        if (dz != 0)
        {
            result = std::make_shared<Transformed>(
                std::move(result), Translation{{0, 0, zs[0] - hh}});
        }

        return result;
    }

    CELER_NOT_IMPLEMENTED("polyhedra");
}

//---------------------------------------------------------------------------//
//! Convert a reflected solid
auto SolidConverter::reflectedsolid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4ReflectedSolid const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("reflectedsolid");
}

//---------------------------------------------------------------------------//
//! Convert a sphere
auto SolidConverter::sphere(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Sphere const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("sphere");
}

//---------------------------------------------------------------------------//
//! Convert a subtraction solid
auto SolidConverter::subtractionsolid(arg_type solid_base) -> result_type
{
    auto vols = this->make_bool_solids(
        dynamic_cast<G4BooleanSolid const&>(solid_base));
    return make_subtraction(std::string{solid_base.GetName()},
                            std::move(vols[0]),
                            std::move(vols[1]));
}

//---------------------------------------------------------------------------//
//! Convert a tessellated solid
auto SolidConverter::tessellatedsolid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4TessellatedSolid const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("tessellatedsolid");
}

//---------------------------------------------------------------------------//
//! Convert a tetrahedron
auto SolidConverter::tet(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Tet const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("tet");
}

//---------------------------------------------------------------------------//
//! Convert a torus
auto SolidConverter::torus(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Torus const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("torus");
}

//---------------------------------------------------------------------------//
//! Convert a generic trapezoid
auto SolidConverter::trap(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Trap const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("trap");
}

//---------------------------------------------------------------------------//
//! Convert a simple trapezoid
auto SolidConverter::trd(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Trd const&>(solid_base);
    CELER_DISCARD(solid);
    CELER_NOT_IMPLEMENTED("trd");
}

//---------------------------------------------------------------------------//
//! Convert a tube section
auto SolidConverter::tubs(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Tubs const&>(solid_base);

    real_type const hh = scale_(solid.GetZHalfLength());
    std::optional<Cylinder> inner;
    if (solid.GetInnerRadius() != 0.0)
    {
        inner = Cylinder{scale_(solid.GetInnerRadius()), hh};
    }

    return make_solid(solid,
                      Cylinder{scale_(solid.GetOuterRadius()), hh},
                      std::move(inner),
                      get_azimuthal_wedge(solid));
}

//---------------------------------------------------------------------------//
//! Convert a union solid
auto SolidConverter::unionsolid(arg_type solid_base) -> result_type
{
    auto vols = this->make_bool_solids(
        dynamic_cast<G4BooleanSolid const&>(solid_base));
    return std::make_shared<AnyObjects>(
        std::string{solid_base.GetName()},
        std::vector{std::move(vols[0]), std::move(vols[1])});
}

//---------------------------------------------------------------------------//
// HELPERS
//---------------------------------------------------------------------------//
//! Create daughter volumes for a boolean solid
auto SolidConverter::make_bool_solids(G4BooleanSolid const& bs)
    -> Array<result_type, 2>
{
    Array<result_type, 2> result;
    for (auto i : range(result.size()))
    {
        G4VSolid const* solid = bs.GetConstituentSolid(i);
        CELER_ASSERT(solid);
        result[i] = (*this)(*solid);
    }
    return result;
}

//---------------------------------------------------------------------------//
//! Calculate the capacity in native celeritas units
double SolidConverter::calc_capacity(G4VSolid const& g4) const
{
    return const_cast<G4VSolid&>(g4).GetCubicVolume() * ipow<3>(scale_(1.0));
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
