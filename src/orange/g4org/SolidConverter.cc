//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/SolidConverter.cc
//---------------------------------------------------------------------------//
#include "SolidConverter.hh"

#include <memory>
#include <typeindex>
#include <typeinfo>  // IWYU pragma: keep
#include <unordered_map>
#include <utility>
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
#include <G4MultiUnion.hh>
#include <G4Orb.hh>
#include <G4Para.hh>
#include <G4Paraboloid.hh>
#include <G4Polycone.hh>
#include <G4Polyhedra.hh>
#include <G4ReflectedSolid.hh>
#include <G4RotationMatrix.hh>
#include <G4ScaledSolid.hh>
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
#include "corecel/io/Repr.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArraySoftUnit.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "orange/orangeinp/CsgObject.hh"
#include "orange/orangeinp/IntersectRegion.hh"
#include "orange/orangeinp/PolySolid.hh"
#include "orange/orangeinp/RevolvedPolygon.hh"
#include "orange/orangeinp/Shape.hh"
#include "orange/orangeinp/Solid.hh"
#include "orange/orangeinp/StackedExtrudedPolygon.hh"
#include "orange/orangeinp/Transformed.hh"
#include "orange/orangeinp/Truncated.hh"

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
 * Get an EnclosedAzi, avoiding values slightly beyond 1 turn.
 *
 * This constructs from native Geant4 radians and truncates to \c real_type,
 * ensuring that roundoff doesn't push the turn beyond a full one.
 */
auto enclosed_azi_radians(double start_rad, double stop_rad)
{
    auto start = native_value_to<RealTurn>(start_rad);
    auto stop = native_value_to<RealTurn>(stop_rad);
    auto delta_turn = value_as<RealTurn>(stop - start);
    CELER_VALIDATE(delta_turn <= 1 || soft_equal(delta_turn, real_type{1}),
                   << "azimuthal restriction [" << start.value() << ", "
                   << stop.value() << "] [turn] exceeds 1 turn");
    if (delta_turn >= real_type{1} || soft_equal(delta_turn, real_type{1}))
    {
        // Avoid roundoff error: return full region
        return EnclosedAzi{};
    }
    return EnclosedAzi{start, stop};
}

//---------------------------------------------------------------------------//
/*!
 * Get an EnclosedPolar, avoiding values slightly beyond a half turn.
 *
 * This constructs from native Geant4 radians and truncates to \c real_type,
 * ensuring that roundoff doesn't push the turn beyond a full one. The
 * G4Sphere::CheckThetaAngles implementation prevents the endpoint being
 * greater than 180 degrees, so we do the same here.
 */
auto enclosed_polar_radians(double start_rad, double stop_rad)
{
    constexpr RealTurn half_turn{0.5};

    auto start = native_value_to<RealTurn>(start_rad);
    auto stop = native_value_to<RealTurn>(stop_rad);
    CELER_VALIDATE(start.value() >= 0 || soft_zero(start.value()),
                   << "polar start angle " << start.value()
                   << " [turn] exceeds half a turn");
    start = min(half_turn, start);
    CELER_VALIDATE(
        stop <= half_turn || soft_equal(stop.value(), half_turn.value()),
        << "polar end angle " << stop.value() << " [turn] exceeds half a turn");
    stop = min(half_turn, stop);
    return EnclosedPolar{start, stop};
}

//---------------------------------------------------------------------------//
/*!
 * Get the enclosed azimuthal angle by a solid.
 *
 * This internally converts from native Geant4 radians.
 */
template<class S>
EnclosedAzi enclosed_azi_from(S const& solid)
{
    auto start = solid.GetStartPhiAngle();
    return enclosed_azi_radians(start, start + solid.GetDeltaPhiAngle());
}

//---------------------------------------------------------------------------//
/*!
 * Get the enclosed azimuthal angle by a G4Torus.
 *
 * This internally converts from native Geant4 radians. This specialization is
 * necessary because Geant4 does not use a consistent API across solids for
 * accessing the start- and delta-phi member variables.
 */
template<>
EnclosedAzi enclosed_azi_from<G4Torus>(G4Torus const& solid)
{
    auto start = solid.GetSPhi();
    return enclosed_azi_radians(start, start + solid.GetDPhi());
}

//---------------------------------------------------------------------------//
/*!
 * Get the enclosed azimuthal angle by a "poly" solid.
 *
 * Geant4 uses different function names for polycone, generic polycone, and
 * polyhedra...
 */
template<class S>
EnclosedAzi enclosed_azi_from_poly(S const& solid)
{
    return enclosed_azi_radians(solid.GetStartPhi(), solid.GetEndPhi());
}

//---------------------------------------------------------------------------//
/*!
 * Get the enclosed polar angle by a solid.
 */
template<class S>
EnclosedPolar enclosed_pol_from(S const& solid)
{
    auto start = solid.GetStartThetaAngle();
    auto delta = solid.GetDeltaThetaAngle();
    return enclosed_polar_radians(start, start + delta);
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
    CELER_EXPECT(is_soft_unit_vector(convert_from_geant(axis)));

    return {native_value_to<Turn>(std::acos(axis.z())),
            atan2turn<real_type>(axis.y(), axis.x())};
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
auto make_shape(G4VSolid const& solid, Args&&... args)
{
    using ::celeritas::orangeinp::make_shape;
    return make_shape<CR>(std::string{solid.GetName()},
                          std::forward<Args>(args)...);
}

//---------------------------------------------------------------------------//
/*!
 * Construct an ORANGE solid using the G4Solid's name and forwarded arguments.
 */
template<class CR, class... Args>
auto make_solid(G4VSolid const& solid, CR&& interior, Args&&... args)
{
    return Solid<CR>::or_shape(std::string{solid.GetName()},
                               std::forward<CR>(interior),
                               std::forward<Args>(args)...);
}

//---------------------------------------------------------------------------//
/*!
 * Construct an ORANGE solid using the G4Solid's name and forwarded arguments.
 */
template<class CR>
auto make_truncated(G4VSolid const& solid,
                    CR&& interior,
                    Truncated::VecPlane&& planes) -> SPConstObject
{
    if (planes.empty())
    {
        return make_shape<CR>(solid, std::forward<CR>(interior));
    }

    return std::make_shared<Truncated>(
        std::string{solid.GetName()},
        std::make_unique<CR>(std::forward<CR>(interior)),
        std::move(planes));
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
    auto radius
        = static_cast<real_type>(std::cbrt(vol / (4.0 / 3.0 * constants::pi)));
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
        SC_TYPE_FUNC(MultiUnion       , multiunion),
        SC_TYPE_FUNC(Orb              , orb),
        SC_TYPE_FUNC(Para             , para),
        SC_TYPE_FUNC(Paraboloid       , paraboloid),
        SC_TYPE_FUNC(Polycone         , polycone),
        SC_TYPE_FUNC(Polyhedra        , polyhedra),
        SC_TYPE_FUNC(ReflectedSolid   , reflectedsolid),
        SC_TYPE_FUNC(ScaledSolid      , scaledsolid),
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

    auto const outer_r = scale_.to<Real2>(solid.GetOuterRadiusMinusZ(),
                                          solid.GetOuterRadiusPlusZ());
    auto const inner_r = scale_.to<Real2>(solid.GetInnerRadiusMinusZ(),
                                          solid.GetInnerRadiusPlusZ());
    auto hh = scale_(solid.GetZHalfLength());

    std::optional<Cone> inner;
    if (any_positive(inner_r))
    {
        inner = Cone{inner_r, hh};
    }

    return make_solid(
        solid, Cone{outer_r, hh}, std::move(inner), enclosed_azi_from(solid));
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
        std::move(daughter), transform_(solid.GetDirectTransform()));
}

//---------------------------------------------------------------------------//
//! Convert an ellipsoid
auto SolidConverter::ellipsoid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Ellipsoid const&>(solid_base);

    auto radii = scale_.to<Real3>(solid.GetSemiAxisMax(to_int(Axis::x)),
                                  solid.GetSemiAxisMax(to_int(Axis::y)),
                                  solid.GetSemiAxisMax(to_int(Axis::z)));

    using Plane = InfPlane;
    std::vector<Plane> truncate;
    if (auto cut = scale_(solid.GetZBottomCut());
        !soft_equal(-radii[to_int(Axis::z)], cut))
    {
        truncate.push_back(InfPlane{Sense::outside, Axis::z, cut});
    }
    if (auto cut = scale_(solid.GetZTopCut());
        !soft_equal(radii[to_int(Axis::z)], cut))
    {
        truncate.push_back(InfPlane{Sense::inside, Axis::z, cut});
    }

    return make_truncated(solid, Ellipsoid{radii}, std::move(truncate));
}

//---------------------------------------------------------------------------//
/*!
 * Convert an elliptical cone.
 *
 * Expressions for lower/upper radii were found by solving the system of
 * equations given by \c G4EllipticalCone:
 *
 * \verbatim
   lower_radii[X]/lower_radii[Y] = upper_radii[X]/upper_radii[Y];
   r_x = (lower_radii[X] - upper_radii[X])/(2 * hh);
   r_y = (lower_radii[Y] - upper_radii[Y])/(2 * hh);
   v = hh * (lower_radii[X] + upper_radii[X])/(lower_radii[X] - upper_radii[X])
 * \endverbatim
 */
auto SolidConverter::ellipticalcone(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4EllipticalCone const&>(solid_base);

    // Read and scale parameters. Do not scale r_x and r_y because they are
    // unitless slopes within the context of this calculation.
    auto r_x = static_cast<real_type>(solid.GetSemiAxisX());
    auto r_y = static_cast<real_type>(solid.GetSemiAxisY());
    auto v = scale_(solid.GetZMax());
    auto hh = scale_(solid.GetZTopCut());

    Real2 lower_radii{r_x * (v + hh), r_y * (v + hh)};
    Real2 upper_radii{r_x * (v - hh), r_y * (v - hh)};

    return make_shape<EllipticalCone>(solid, lower_radii, upper_radii, hh);
}

//---------------------------------------------------------------------------//
//! Convert an elliptical tube
auto SolidConverter::ellipticaltube(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4EllipticalTube const&>(solid_base);

    auto rx = scale_(solid.GetDx());
    auto ry = scale_(solid.GetDy());
    auto halfheight = scale_(solid.GetDz());

    return make_shape<EllipticalCylinder>(solid, Real2({rx, ry}), halfheight);
}

//---------------------------------------------------------------------------//
//! Convert an extruded solid
auto SolidConverter::extrudedsolid(arg_type solid_base) -> result_type
{
    using VecReal = StackedExtrudedPolygon::VecReal;
    using VecReal2 = StackedExtrudedPolygon::VecReal2;
    using VecReal3 = StackedExtrudedPolygon::VecReal3;

    auto const& solid = dynamic_cast<G4ExtrudedSolid const&>(solid_base);

    // Get the polygon and reverse its order; ORANGE uses standard
    // counterclockwise ordering for polygons whereas GEANT4 uses clockwise
    // ordering.
    std::vector<G4TwoVector> g4polygon = solid.GetPolygon();
    VecReal2 polygon;
    for (auto i : range<int>(g4polygon.size()).step(-1))
    {
        auto point = g4polygon[i];
        polygon.push_back(Real2{scale_(point[0]), scale_(point[1])});
    }

    // Construct polyline and scaling
    VecReal3 polyline;
    VecReal scaling;
    polyline.reserve(solid.GetNofZSections());
    scaling.reserve(solid.GetNofZSections());
    for (auto const& z_section : solid.GetZSections())
    {
        polyline.push_back({
            scale_(z_section.fOffset[0]),
            scale_(z_section.fOffset[1]),
            scale_(z_section.fZ),
        });
        scaling.push_back(z_section.fScale);
    }

    return StackedExtrudedPolygon::or_solid(std::string{solid.GetName()},
                                            std::move(polygon),
                                            std::move(polyline),
                                            std::move(scaling));
}

//---------------------------------------------------------------------------//
//! Convert a generic polycone
auto SolidConverter::genericpolycone(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4GenericPolycone const&>(solid_base);

    // Get the polygon. Although Geant4 prefers clockwise order upon input,
    // GetCorner actually returns points in counterclockwise order, as used
    // by ORANGE.
    size_type num_points = solid.GetNumRZCorner();
    std::vector<Real2> polygon;
    polygon.reserve(num_points);
    for (auto i : range(num_points))
    {
        auto point = solid.GetCorner(i);
        polygon.push_back(scale_.to<Real2>(point.r, point.z));
    }

    return std::make_shared<RevolvedPolygon>(std::string{solid.GetName()},
                                             std::move(polygon),
                                             enclosed_azi_from_poly(solid));
}

//---------------------------------------------------------------------------//
//! Convert a generic trapezoid
auto SolidConverter::generictrap(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4GenericTrap const&>(solid_base);

    auto const& vtx = solid.GetVertices();
    CELER_ASSERT(vtx.size() == 8);

    std::vector<Real2> lower(4), upper(4);
    for (auto i : range(4))
    {
        lower[i] = scale_.to<Real2>(vtx[i].x(), vtx[i].y());
        upper[i] = scale_.to<Real2>(vtx[i + 4].x(), vtx[i + 4].y());
    }
    real_type hh = scale_(solid.GetZHalfLength());

    return make_shape<GenPrism>(solid, hh, std::move(lower), std::move(upper));
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
//! Convert a multiunion
auto SolidConverter::multiunion(arg_type solid_base) -> result_type
{
    auto const& mu = dynamic_cast<G4MultiUnion const&>(solid_base);
    auto n = mu.GetNumberOfSolids();
    std::vector<result_type> vols(n);

    for (auto i : range(n))
    {
        auto vol = (*this)(*(mu.GetSolid(i)));
        vols[i] = std::make_shared<Transformed>(
            std::move(vol), transform_(mu.GetTransformation(i)));
    }

    return std::make_shared<AnyObjects>(std::string{solid_base.GetName()},
                                        std::move(vols));
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

    auto lower_radius = scale_(solid.GetRadiusMinusZ());
    auto upper_radius = scale_(solid.GetRadiusPlusZ());
    auto hh = scale_(solid.GetZHalfLength());

    return make_shape<Paraboloid>(solid, lower_radius, upper_radius, hh);
}

//---------------------------------------------------------------------------//
//! Convert a polycone
auto SolidConverter::polycone(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Polycone const&>(solid_base);
    auto const& params = *solid.GetOriginalParameters();

    std::vector<real_type> z(params.Num_z_planes);
    std::vector<real_type> rmin(z.size());
    std::vector<real_type> rmax(z.size());
    for (auto i : range(z.size()))
    {
        z[i] = scale_(params.Z_values[i]);
        rmin[i] = scale_(params.Rmin[i]);
        rmax[i] = scale_(params.Rmax[i]);
    }

    if (!any_positive(rmin))
    {
        // No interior shape
        rmin.clear();
    }

    if (!z.empty() && z.front() > z.back())
    {
        CELER_LOG(warning) << "Polycone '" << solid.GetName()
                           << "' z coordinates are out of order: " << repr(z);
    }

    return PolyCone::or_solid(
        std::string{solid.GetName()},
        PolySegments{std::move(rmin), std::move(rmax), std::move(z)},
        enclosed_azi_from_poly(solid));
}

//---------------------------------------------------------------------------//
//! Convert a polyhedron
auto SolidConverter::polyhedra(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Polyhedra const&>(solid_base);
    auto const& params = *solid.GetOriginalParameters();

    // Convert from circumradius to apothem
    double const radius_factor = cospi(1 / static_cast<double>(params.numSide));

    std::vector<real_type> zs(params.Num_z_planes);
    std::vector<real_type> rmin(zs.size());
    std::vector<real_type> rmax(zs.size());
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

    // Get orientation from the start/end phi, which still may be a full Turn
    auto frac_turn = native_value_to<Turn>(solid.GetStartPhi()).value();
    double const orientation
        = std::fmod(params.numSide * frac_turn, real_type{1});

    return PolyPrism::or_solid(
        std::string{solid.GetName()},
        PolySegments{std::move(rmin), std::move(rmax), std::move(zs)},
        enclosed_azi_from_poly(solid),
        params.numSide,
        orientation);
}

//---------------------------------------------------------------------------//
//! Convert a reflected solid
auto SolidConverter::reflectedsolid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4ReflectedSolid const&>(solid_base);
    G4VSolid* underlying = solid.GetConstituentMovedSolid();
    CELER_ASSERT(underlying);

    // Convert unreflected solid
    auto converted = (*this)(*underlying);

    // Add a reflecting transform
    return std::make_shared<Transformed>(
        std::move(converted), transform_(solid.GetDirectTransform3D()));
}

//---------------------------------------------------------------------------//
//! Convert a scaled solid
auto SolidConverter::scaledsolid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4ScaledSolid const&>(solid_base);
    G4VSolid* underlying = solid.GetUnscaledSolid();
    CELER_ASSERT(underlying);

    // Convert unscaled solid
    auto converted = (*this)(*underlying);

    // Add a scaling transform
    return std::make_shared<Transformed>(
        std::move(converted), transform_(solid.GetScaleTransform()));
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

    auto polar_cone = enclosed_pol_from(solid);

    return make_solid(solid,
                      Sphere{scale_(solid.GetOuterRadius())},
                      std::move(inner),
                      enclosed_azi_from(solid),
                      std::move(polar_cone));
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
    std::vector<G4ThreeVector> vertices = solid.GetVertices();
    CELER_ASSERT(vertices.size() == 4);
    return make_shape<Tet>(
        solid,
        scale_.to<Tet::ArrReal3>(
            vertices[0], vertices[1], vertices[2], vertices[3]));
}

//---------------------------------------------------------------------------//
//! Convert a torus
auto SolidConverter::torus(arg_type solid_base) -> result_type
{
    CELER_LOG(warning) << "G4Torus is not fully supported; approximating with "
                          "bounding cylinders";
    auto const& solid = dynamic_cast<G4Torus const&>(solid_base);
    auto rmax = scale_(solid.GetRmax());
    auto rtor = scale_(solid.GetRtor());

    std::optional<Cylinder> inner{std::in_place, rtor - rmax, rmax};

    return make_solid(solid,
                      Cylinder{rtor + rmax, rmax},
                      std::move(inner),
                      enclosed_azi_from(solid));
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

    GenPrism::TrapFace lo;
    lo.hy = scale_(solid.GetYHalfLength1());
    lo.hx_lo = scale_(solid.GetXHalfLength1());
    lo.hx_hi = scale_(solid.GetXHalfLength2());
    lo.alpha = native_value_to<Turn>(alpha_1);

    GenPrism::TrapFace hi;
    hi.hy = scale_(solid.GetYHalfLength2());
    hi.hx_lo = scale_(solid.GetXHalfLength3());
    hi.hx_hi = scale_(solid.GetXHalfLength4());
    hi.alpha = native_value_to<Turn>(alpha_2);

    return make_shape<GenPrism>(solid,
                                GenPrism::from_trap(hz, theta, phi, lo, hi));
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

    return make_shape<GenPrism>(
        solid, GenPrism::from_trd(hz, {hx1, hy1}, {hx2, hy2}));
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
                      enclosed_azi_from(solid));
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
    return const_cast<G4VSolid&>(g4).GetCubicVolume() * ipow<3>(scale_.value());
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
