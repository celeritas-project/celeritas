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
#include "corecel/math/ArraySoftUnit.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "orange/orangeinp/CsgObject.hh"
#include "orange/orangeinp/PolySolid.hh"
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
 * Get the enclosed azimuthal angle by a solid.
 *
 * This internally converts from native Geant4 radians.
 */
template<class S>
SolidEnclosedAngle make_wedge_azimuthal(S const& solid)
{
    return SolidEnclosedAngle{native_value_to<Turn>(solid.GetStartPhiAngle()),
                              native_value_to<Turn>(solid.GetDeltaPhiAngle())};
}

//---------------------------------------------------------------------------//
/*!
 * Get the enclosed azimuthal angle by a "poly" solid.
 *
 * Geant4 uses different function names for polycone, generic polycone, and
 * polyhedra...
 */
template<class S>
SolidEnclosedAngle make_wedge_azimuthal_poly(S const& solid)
{
    auto start = native_value_to<Turn>(solid.GetStartPhi());
    auto stop = native_value_to<Turn>(solid.GetEndPhi());
    return SolidEnclosedAngle{start, stop - start};
}

//---------------------------------------------------------------------------//
/*!
 * Get the enclosed polar angle by a solid.
 *
 * This internally converts from native Geant4 radians.
 */
template<class S>
SolidEnclosedAngle make_wedge_polar(S const& solid)
{
    return SolidEnclosedAngle{
        native_value_to<Turn>(solid.GetStartThetaAngle()),
        native_value_to<Turn>(solid.GetDeltaThetaAngle())};
}

//---------------------------------------------------------------------------//
/*!
 * Return theta, phi angles for a G4Para or G4Trap given their symmetry axis.
 *
 * Certain Geant4 shapes are constructed by skewing the z axis and providing
 * the polar/azimuthal angle of the transformed axis. This calculates that
 * transform by converting from cartesian to spherical coordinates.
 *
 * The components of the symmetry axis for G4Para/Trap are always encoded as a
 * vector
 * \f$ (\mu \tan(\theta)\cos(\phi), \mu \tan(\theta)\sin(\phi), \mu) \f$.
 */
[[maybe_unused]] auto to_polar(G4ThreeVector const& axis)
    -> std::pair<Turn, Turn>
{
    CELER_EXPECT(axis.z() > 0);
    CELER_EXPECT(
        is_soft_unit_vector(Array<double, 3>{axis.x(), axis.y(), axis.z()}));

    double const theta = std::acos(axis.z());
    double const phi = std::atan2(axis.y(), axis.x());
    return {native_value_to<Turn>(theta), native_value_to<Turn>(phi)};
}

//---------------------------------------------------------------------------//
/*!
 * Return theta, phi angles for a G4Para or G4Trap given their symmetry axis.
 *
 * Certain Geant4 shapes are constructed by skewing the z axis and providing
 * the polar/azimuthal angle of the transformed axis. This calculates that
 * transform by converting from cartesian to spherical coordinates.
 *
 * The components of the symmetry axis for G4Para/Trap are always encoded as a
 * vector \f$ (A \tan(\theta)\cos(\phi), A \tan(\theta)\sin(phi), A) \f$.
 */
template<class S>
auto calculate_theta_phi(S const& solid) -> std::pair<Turn, Turn>
{
#if G4VERSION_NUMBER >= 1100
    double const theta = solid.GetTheta();
    double const phi = solid.GetPhi();
    return {native_value_to<Turn>(theta), native_value_to<Turn>(phi)};
#else
    return to_polar(solid.GetSymAxis());
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Construct a shape using the solid's name and forwarded arguments.
 */
template<class CR, class... Args>
auto make_shape(std::string&& name, Args&&... args)
{
    return std::make_shared<Shape<CR>>(std::move(name),
                                       CR{std::forward<Args>(args)...});
}

//---------------------------------------------------------------------------//
/*!
 * Construct a shape using the solid's name and forwarded arguments.
 */
template<class CR, class... Args>
auto make_shape(G4VSolid const& solid, Args&&... args)
{
    return make_shape<CR>(std::string{solid.GetName()},
                          std::forward<Args>(args)...);
}

//---------------------------------------------------------------------------//
/*!
 * Construct an ORANGE solid using the G4Solid's name and forwarded arguments.
 */
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
template<class Container>
bool any_positive(Container const& c)
{
    return std::any_of(c.begin(), c.end(), [](auto r) { return r > 0; });
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
 * Convert a Geant4 solid to a sphere with equivalent capacity.
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

    std::optional<Cone> inner;
    if (any_positive(inner_r))
    {
        inner = Cone{inner_r, hh};
    }

    return make_solid(
        solid, Cone{outer_r, hh}, std::move(inner), make_wedge_azimuthal(solid));
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

    // Note that GetDirectTransform is an affine transform that combines the
    // daughter-to-parent ("object") translation with an inverted
    // [parent-to-daughter, "frame"] rotation
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

    auto const& vtx = solid.GetVertices();
    CELER_ASSERT(vtx.size() == 8);

    std::vector<GenTrap::Real2> lower(4), upper(4);
    for (auto i : range(4))
    {
        lower[i] = scale_.to<GenTrap::Real2>(vtx[i].x(), vtx[i].y());
        upper[i] = scale_.to<GenTrap::Real2>(vtx[i + 4].x(), vtx[i + 4].y());
    }
    real_type hh = scale_(solid.GetZHalfLength());

    return make_shape<GenTrap>(solid, hh, std::move(lower), std::move(upper));
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
    auto const [theta, phi] = calculate_theta_phi(solid);
    return make_shape<Parallelepiped>(
        solid,
        scale_.to<Real3>(solid.GetXHalfLength(),
                         solid.GetYHalfLength(),
                         solid.GetZHalfLength()),
        native_value_to<Turn>(std::atan(solid.GetTanAlpha())),
        theta,
        phi);
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

    if (!any_positive(rmin))
    {
        // No interior shape
        rmin.clear();
    }

    return PolyCone::or_solid(
        std::string{solid.GetName()},
        PolySegments{std::move(rmin), std::move(rmax), std::move(zs)},
        make_wedge_azimuthal_poly(solid));
}

//---------------------------------------------------------------------------//
//! Convert a polyhedron
auto SolidConverter::polyhedra(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Polyhedra const&>(solid_base);
    auto const& params = *solid.GetOriginalParameters();

    // Convert from circumradius to apothem
    double const radius_factor = std::cos(m_pi / params.numSide);

    std::vector<double> zs(params.Num_z_planes);
    std::vector<double> rmin(zs.size());
    std::vector<double> rmax(zs.size());
    for (auto i : range(zs.size()))
    {
        zs[i] = scale_(params.Z_values[i]);
        rmin[i] = scale_(params.Rmin[i]) * radius_factor;
        rmax[i] = scale_(params.Rmax[i]) * radius_factor;
    }

    if (!any_positive(rmin))
    {
        // No interior shape
        rmin.clear();
    }

    auto angle = make_wedge_azimuthal_poly(solid);
    double const orientation
        = std::fmod(params.numSide * angle.start().value(), real_type{1});

    return PolyPrism::or_solid(
        std::string{solid.GetName()},
        PolySegments{std::move(rmin), std::move(rmax), std::move(zs)},
        std::move(angle),
        params.numSide,
        orientation);
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
    std::optional<Sphere> inner;
    if (double inner_r = solid.GetInnerRadius())
    {
        inner = Sphere{scale_(inner_r)};
    }

    auto polar_wedge = make_wedge_polar(solid);
    if (!soft_equal(value_as<Turn>(polar_wedge.interior()), 0.5))
    {
        CELER_NOT_IMPLEMENTED("sphere with polar limits");
    }

    return make_solid(solid,
                      Sphere{scale_(solid.GetOuterRadius())},
                      std::move(inner),
                      make_wedge_azimuthal(solid));
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
/*!
 * Convert a trapezoid.
 *
 * Note that the numbers of x,y,z parameters in the G4Trap are related to the
 * fact that the two z-faces are parallel (separated by hz) and the 4 x-wedges
 * (2 in each z-face) are also parallel (separated by hy1,2).
 *
 * Reference:
 * https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/Detector/Geometry/geomSolids.html#constructed-solid-geometry-csg-solids
 */
auto SolidConverter::trap(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Trap const&>(solid_base);

    auto const [theta, phi] = calculate_theta_phi(solid);
#if G4VERSION_NUMBER >= 1100
    double const alpha_1 = solid.GetAlpha1();
    double const alpha_2 = solid.GetAlpha2();
#else
    double const alpha_1 = std::atan(solid.GetTanAlpha1());
    double const alpha_2 = std::atan(solid.GetTanAlpha2());
#endif

    auto hz = scale_(solid.GetZHalfLength());

    GenTrap::TrapFace lo;
    lo.hy = scale_(solid.GetYHalfLength1());
    lo.hx_lo = scale_(solid.GetXHalfLength1());
    lo.hx_hi = scale_(solid.GetXHalfLength2());
    lo.tan_alpha = alpha_1;

    GenTrap::TrapFace hi;
    hi.hy = scale_(solid.GetYHalfLength2());
    hi.hx_lo = scale_(solid.GetXHalfLength3());
    hi.hx_hi = scale_(solid.GetXHalfLength4());
    hi.tan_alpha = alpha_2;

    return make_shape<GenTrap>(solid,
                               GenTrap::from_trap(hz, theta, phi, lo, hi));
}

//---------------------------------------------------------------------------//
//! Convert a simple trapezoid
auto SolidConverter::trd(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Trd const&>(solid_base);

    auto hz = scale_(solid.GetZHalfLength());
    auto hy1 = scale_(solid.GetYHalfLength1());
    auto hy2 = scale_(solid.GetYHalfLength2());
    auto hx1 = scale_(solid.GetXHalfLength1());
    auto hx2 = scale_(solid.GetXHalfLength2());

    return make_shape<GenTrap>(solid,
                               GenTrap::from_trd(hz, {hx1, hy1}, {hx2, hy2}));
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
                      make_wedge_azimuthal(solid));
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
