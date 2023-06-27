//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/g4vg/SolidConverter.cc
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
#include <G4GDMLWriteStructure.hh>
#include <G4GenericPolycone.hh>
#include <G4GenericTrap.hh>
#include <G4Hype.hh>
#include <G4IntersectionSolid.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4Navigator.hh>
#include <G4Orb.hh>
#include <G4PVDivision.hh>
#include <G4PVParameterised.hh>
#include <G4PVReplica.hh>
#include <G4Para.hh>
#include <G4Paraboloid.hh>
#include <G4Polycone.hh>
#include <G4Polyhedra.hh>
#include <G4PropagatorInField.hh>
#include <G4ReflectedSolid.hh>
#include <G4ReflectionFactory.hh>
#include <G4RotationMatrix.hh>
#include <G4Sphere.hh>
#include <G4SubtractionSolid.hh>
#include <G4TessellatedSolid.hh>
#include <G4Tet.hh>
#include <G4ThreeVector.hh>
#include <G4Torus.hh>
#include <G4Transform3D.hh>
#include <G4Trap.hh>
#include <G4Trd.hh>
#include <G4Tubs.hh>
#include <G4UnionSolid.hh>
#include <G4VSolid.hh>
#include <G4Version.hh>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/UnplacedAssembly.h>
#include <VecGeom/volumes/UnplacedBooleanVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>
#include <VecGeom/volumes/UnplacedCone.h>
#include <VecGeom/volumes/UnplacedCutTube.h>
#include <VecGeom/volumes/UnplacedEllipsoid.h>
#include <VecGeom/volumes/UnplacedEllipticalCone.h>
#include <VecGeom/volumes/UnplacedEllipticalTube.h>
#include <VecGeom/volumes/UnplacedGenTrap.h>
#include <VecGeom/volumes/UnplacedGenericPolycone.h>
#include <VecGeom/volumes/UnplacedHype.h>
#include <VecGeom/volumes/UnplacedOrb.h>
#include <VecGeom/volumes/UnplacedParaboloid.h>
#include <VecGeom/volumes/UnplacedParallelepiped.h>
#include <VecGeom/volumes/UnplacedPolycone.h>
#include <VecGeom/volumes/UnplacedPolyhedron.h>
#include <VecGeom/volumes/UnplacedSExtruVolume.h>
#include <VecGeom/volumes/UnplacedScaledShape.h>
#include <VecGeom/volumes/UnplacedSphere.h>
#include <VecGeom/volumes/UnplacedTessellated.h>
#include <VecGeom/volumes/UnplacedTet.h>
#include <VecGeom/volumes/UnplacedTorus2.h>
#include <VecGeom/volumes/UnplacedTrapezoid.h>
#include <VecGeom/volumes/UnplacedTrd.h>
#include <VecGeom/volumes/UnplacedTube.h>

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "celeritas/Constants.hh"

#include "Scaler.hh"
#include "Transformer.hh"

namespace
{
/*!
 * Return theta, phi angles for a G4Para or G4Trap given their symmetry axis
 */
auto calculate_theta_phi(G4ThreeVector const& axis) -> std::pair<double, double>
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
}  // namespace

using namespace vecgeom;

namespace celeritas
{
namespace g4vg
{
//---------------------------------------------------------------------------//
/*!
 * Convert a geant4 solid to a VecGeom "unplaced volume".
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
    return GeoManager::MakeInstance<UnplacedOrb>(radius);
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
    #define VGSC_TYPE_FUNC(MIXED, LOWER) \
    {std::type_index(typeid(G4##MIXED)), &SolidConverter::LOWER}
    static const MapTypeConverter type_to_converter = {
        VGSC_TYPE_FUNC(Box              , box),
        VGSC_TYPE_FUNC(Cons             , cons),
        VGSC_TYPE_FUNC(CutTubs          , cuttubs),
        VGSC_TYPE_FUNC(Ellipsoid        , ellipsoid),
        VGSC_TYPE_FUNC(EllipticalCone   , ellipticalcone),
        VGSC_TYPE_FUNC(EllipticalTube   , ellipticaltube),
        VGSC_TYPE_FUNC(ExtrudedSolid    , extrudedsolid),
        VGSC_TYPE_FUNC(GenericPolycone  , genericpolycone),
        VGSC_TYPE_FUNC(GenericTrap      , generictrap),
        VGSC_TYPE_FUNC(Hype             , hype),
        VGSC_TYPE_FUNC(IntersectionSolid, intersectionsolid),
        VGSC_TYPE_FUNC(Orb              , orb),
        VGSC_TYPE_FUNC(Para             , para),
        VGSC_TYPE_FUNC(Paraboloid       , paraboloid),
        VGSC_TYPE_FUNC(Polycone         , polycone),
        VGSC_TYPE_FUNC(Polyhedra        , polyhedra),
        VGSC_TYPE_FUNC(ReflectedSolid   , reflectedsolid),
        VGSC_TYPE_FUNC(Sphere           , sphere),
        VGSC_TYPE_FUNC(SubtractionSolid , subtractionsolid),
        VGSC_TYPE_FUNC(TessellatedSolid , tessellatedsolid),
        VGSC_TYPE_FUNC(Tet              , tet),
        VGSC_TYPE_FUNC(Torus            , torus),
        VGSC_TYPE_FUNC(Trap             , trap),
        VGSC_TYPE_FUNC(Trd              , trd),
        VGSC_TYPE_FUNC(Tubs             , tubs),
        VGSC_TYPE_FUNC(UnionSolid       , unionsolid),
    };
    // clang-format on
#undef VGSC_TYPE_FUNC

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
    if (CELER_UNLIKELY(compare_volumes_))
    {
        CELER_ASSERT(result);
        this->compare_volumes(solid_base, *result);
    }

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
    return GeoManager::MakeInstance<UnplacedBox>(
        this->convert_scale_(solid.GetXHalfLength()),
        this->convert_scale_(solid.GetYHalfLength()),
        this->convert_scale_(solid.GetZHalfLength()));
}

//---------------------------------------------------------------------------//
//! Convert a cone section
auto SolidConverter::cons(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Cons const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedCone>(
        this->convert_scale_(solid.GetInnerRadiusMinusZ()),
        this->convert_scale_(solid.GetOuterRadiusMinusZ()),
        this->convert_scale_(solid.GetInnerRadiusPlusZ()),
        this->convert_scale_(solid.GetOuterRadiusPlusZ()),
        this->convert_scale_(solid.GetZHalfLength()),
        solid.GetStartPhiAngle(),
        solid.GetDeltaPhiAngle());
}

//---------------------------------------------------------------------------//
//! Convert a cut tube
auto SolidConverter::cuttubs(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4CutTubs const&>(solid_base);
    G4ThreeVector lowNorm = solid.GetLowNorm();
    G4ThreeVector hiNorm = solid.GetHighNorm();
    return GeoManager::MakeInstance<UnplacedCutTube>(
        this->convert_scale_(solid.GetInnerRadius()),
        this->convert_scale_(solid.GetOuterRadius()),
        this->convert_scale_(solid.GetZHalfLength()),
        solid.GetStartPhiAngle(),
        solid.GetDeltaPhiAngle(),
        Vector3D<Precision>(lowNorm[0], lowNorm[1], lowNorm[2]),
        Vector3D<Precision>(hiNorm[0], hiNorm[1], hiNorm[2]));
}

//---------------------------------------------------------------------------//
//! Convert an ellipsoid
auto SolidConverter::ellipsoid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Ellipsoid const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedEllipsoid>(
        this->convert_scale_(solid.GetDx()),
        this->convert_scale_(solid.GetDy()),
        this->convert_scale_(solid.GetDz()),
        this->convert_scale_(solid.GetZBottomCut()),
        this->convert_scale_(solid.GetZTopCut()));
}

//---------------------------------------------------------------------------//
//! Convert an elliptical cone
auto SolidConverter::ellipticalcone(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4EllipticalCone const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedEllipticalCone>(
        solid.GetSemiAxisX(),
        solid.GetSemiAxisY(),
        this->convert_scale_(solid.GetZMax()),
        this->convert_scale_(solid.GetZTopCut()));
}

//---------------------------------------------------------------------------//
//! Convert an elliptical tube
auto SolidConverter::ellipticaltube(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4EllipticalTube const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedEllipticalTube>(
        this->convert_scale_(solid.GetDx()),
        this->convert_scale_(solid.GetDy()),
        this->convert_scale_(solid.GetDz()));
}

//---------------------------------------------------------------------------//
//! Convert an extruded solid
auto SolidConverter::extrudedsolid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4ExtrudedSolid const&>(solid_base);

    // Convert vertices
    std::vector<double> x(solid.GetNofVertices());
    std::vector<double> y(x.size());
    for (auto i : range(x.size()))
    {
        std::tie(x[i], y[i]) = this->convert_scale_(solid.GetVertex(i));
    }

    // Convert Z sections
    std::vector<double> z(solid.GetNofZSections());
    if (z.size() != 2)
    {
        CELER_LOG(error) << "Extruded solid named '" << solid_base.GetName()
                         << "' has " << z.size()
                         << " Z sections, but VecGeom requires exactly 2";
        CELER_ASSERT(z.size() >= 2);
    }
    for (auto i : range(z.size()))
    {
        G4ExtrudedSolid::ZSection const& zsec = solid.GetZSection(i);
        CELER_VALIDATE(zsec.fScale == 1.0,
                       << "unsupported scale factor '" << zsec.fScale << '\'');
        CELER_VALIDATE(zsec.fOffset.x() == 0.0 && zsec.fOffset.y() == 0.0,
                       << "unsupported z section translation ("
                       << zsec.fOffset.x() << "," << zsec.fOffset.y() << ")");
        z[i] = this->convert_scale_(zsec.fZ);
    }

    return GeoManager::MakeInstance<UnplacedSExtruVolume>(
        x.size(), x.data(), y.data(), z.front(), z.back());
}

//---------------------------------------------------------------------------//
//! Convert a generic polycone
auto SolidConverter::genericpolycone(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4GenericPolycone const&>(solid_base);

    std::vector<double> zs(solid.GetNumRZCorner());
    std::vector<double> rs(zs.size());
    for (auto i : range(zs.size()))
    {
        G4PolyconeSideRZ const& rzCorner = solid.GetCorner(i);
        zs[i] = this->convert_scale_(rzCorner.z);
        rs[i] = this->convert_scale_(rzCorner.r);
    }

    return GeoManager::MakeInstance<UnplacedGenericPolycone>(
        solid.GetStartPhi(),
        solid.GetEndPhi() - solid.GetStartPhi(),
        rs.size(),
        rs.data(),
        zs.data());
}

//---------------------------------------------------------------------------//
//! Convert a generic trapezoid
auto SolidConverter::generictrap(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4GenericTrap const&>(solid_base);

    std::vector<double> vx(solid.GetNofVertices());
    std::vector<double> vy(vx.size());
    for (auto i : range(vx.size()))
    {
        std::tie(vx[i], vy[i]) = this->convert_scale_(solid.GetVertex(i));
    }

    return GeoManager::MakeInstance<UnplacedGenTrap>(
        vx.data(), vy.data(), this->convert_scale_(solid.GetZHalfLength()));
}

//---------------------------------------------------------------------------//
//! Convert a hyperbola
auto SolidConverter::hype(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Hype const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedHype>(
        this->convert_scale_(solid.GetInnerRadius()),
        this->convert_scale_(solid.GetOuterRadius()),
        solid.GetInnerStereo(),
        solid.GetOuterStereo(),
        this->convert_scale_(solid.GetZHalfLength()));
}

//---------------------------------------------------------------------------//
//! Convert an intersection solid
auto SolidConverter::intersectionsolid(arg_type solid_base) -> result_type
{
    PlacedBoolVolumes pv = this->convert_bool_impl(
        static_cast<G4BooleanSolid const&>(solid_base));
    return GeoManager::MakeInstance<UnplacedBooleanVolume<kIntersection>>(
        kSubtraction, pv[0], pv[1]);
}

//---------------------------------------------------------------------------//
//! Convert an orb
auto SolidConverter::orb(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Orb const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedOrb>(
        this->convert_scale_(solid.GetRadius()));
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
    // Theta/Phi are not directly accessible before 11.0 but are encoded in the
    // symmetry axis
    auto const [theta, phi] = calculate_theta_phi(solid.GetSymAxis());
#endif
    return GeoManager::MakeInstance<UnplacedParallelepiped>(
        this->convert_scale_(solid.GetXHalfLength()),
        this->convert_scale_(solid.GetYHalfLength()),
        this->convert_scale_(solid.GetZHalfLength()),
        std::atan(solid.GetTanAlpha()),
        theta,
        phi);
}

//---------------------------------------------------------------------------//
//! Convert a paraboloid
auto SolidConverter::paraboloid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Paraboloid const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedParaboloid>(
        this->convert_scale_(solid.GetRadiusMinusZ()),
        this->convert_scale_(solid.GetRadiusPlusZ()),
        this->convert_scale_(solid.GetZHalfLength()));
}

//---------------------------------------------------------------------------//
//! Convert a polycone
auto SolidConverter::polycone(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Polycone const&>(solid_base);
    auto const& params = *solid.GetOriginalParameters();

    std::vector<double> zvals(params.Num_z_planes);
    std::vector<double> rmins(zvals.size());
    std::vector<double> rmaxs(zvals.size());
    for (auto i : range(zvals.size()))
    {
        zvals[i] = this->convert_scale_(params.Z_values[i]);
        rmins[i] = this->convert_scale_(params.Rmin[i]);
        rmaxs[i] = this->convert_scale_(params.Rmax[i]);
    }
    return GeoManager::MakeInstance<UnplacedPolycone>(params.Start_angle,
                                                      params.Opening_angle,
                                                      zvals.size(),
                                                      zvals.data(),
                                                      rmins.data(),
                                                      rmaxs.data());
}

//---------------------------------------------------------------------------//
//! Convert a polyhedron
auto SolidConverter::polyhedra(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Polyhedra const&>(solid_base);
    auto const& params = *solid.GetOriginalParameters();
    // G4 has a different radius conventions (than TGeo, gdml, VecGeom)!
    double const radius_factor
        = std::cos(0.5 * params.Opening_angle / params.numSide);

    std::vector<double> zs(params.Num_z_planes);
    std::vector<double> rmins(zs.size());
    std::vector<double> rmaxs(zs.size());
    for (auto i : range(zs.size()))
    {
        zs[i] = this->convert_scale_(params.Z_values[i]);
        rmins[i] = this->convert_scale_(params.Rmin[i] * radius_factor);
        rmaxs[i] = this->convert_scale_(params.Rmax[i] * radius_factor);
    }

    auto phistart = std::fmod(params.Start_angle, 2 * constants::pi);

    return GeoManager::MakeInstance<UnplacedPolyhedron>(phistart,
                                                        params.Opening_angle,
                                                        params.numSide,
                                                        zs.size(),
                                                        zs.data(),
                                                        rmins.data(),
                                                        rmaxs.data());
}

//---------------------------------------------------------------------------//
//! Convert a reflected solid
auto SolidConverter::reflectedsolid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4ReflectedSolid const&>(solid_base);
    G4VSolid* underlying = solid.GetConstituentMovedSolid();
    CELER_ASSERT(underlying);
    return (*this)(*underlying);
}

//---------------------------------------------------------------------------//
//! Convert a sphere
auto SolidConverter::sphere(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Sphere const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedSphere>(
        this->convert_scale_(solid.GetInnerRadius()),
        this->convert_scale_(solid.GetOuterRadius()),
        solid.GetStartPhiAngle(),
        solid.GetDeltaPhiAngle(),
        solid.GetStartThetaAngle(),
        solid.GetDeltaThetaAngle());
}

//---------------------------------------------------------------------------//
//! Convert a subtraction solid
auto SolidConverter::subtractionsolid(arg_type solid_base) -> result_type
{
    PlacedBoolVolumes pv = this->convert_bool_impl(
        static_cast<G4BooleanSolid const&>(solid_base));
    return GeoManager::MakeInstance<UnplacedBooleanVolume<kSubtraction>>(
        kSubtraction, pv[0], pv[1]);
}

//---------------------------------------------------------------------------//
//! Convert a tessellated solid
auto SolidConverter::tessellatedsolid(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4TessellatedSolid const&>(solid_base);
    using Vertex = vecgeom::Vector3D<vecgeom::Precision>;

    auto* result = GeoManager::MakeInstance<UnplacedTessellated>();

    for (auto i : range(solid.GetNumberOfFacets()))
    {
        G4VFacet const& facet = *solid.GetFacet(i);
        int const num_vtx = facet.GetNumberOfVertices();
        Array<Vertex, 4> vtx;
        for (auto iv : range(num_vtx))
        {
            auto vxg4 = facet.GetVertex(iv);
            vtx[iv].Set(this->convert_scale_(vxg4.x()),
                        this->convert_scale_(vxg4.y()),
                        this->convert_scale_(vxg4.z()));
        }

        if (num_vtx == 3)
        {
            result->AddTriangularFacet(vtx[0], vtx[1], vtx[2], ABSOLUTE);
        }
        else
        {
            result->AddQuadrilateralFacet(
                vtx[0], vtx[1], vtx[2], vtx[3], ABSOLUTE);
        }
    }
    result->Close();
    return result;
}

//---------------------------------------------------------------------------//
//! Convert a tetrahedron
auto SolidConverter::tet(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Tet const&>(solid_base);
    G4ThreeVector anchor, p1, p2, p3;
    solid.GetVertices(anchor, p1, p2, p3);
    return GeoManager::MakeInstance<UnplacedTet>(this->convert_scale_(anchor),
                                                 this->convert_scale_(p1),
                                                 this->convert_scale_(p2),
                                                 this->convert_scale_(p3));
}

//---------------------------------------------------------------------------//
//! Convert a torus
auto SolidConverter::torus(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Torus const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedTorus2>(
        this->convert_scale_(solid.GetRmin()),
        this->convert_scale_(solid.GetRmax()),
        this->convert_scale_(solid.GetRtor()),
        solid.GetSPhi(),
        solid.GetDPhi());
}

//---------------------------------------------------------------------------//
//! Convert a generic trapezoid
auto SolidConverter::trap(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Trap const&>(solid_base);
#if G4VERSION_NUMBER >= 1100
    double const theta = solid.GetTheta();
    double const phi = solid.GetPhi();
    double const alpha_1 = solid.GetAlpha1();
    double const alpha_2 = solid.GetAlpha2();
#else
    // Theta/Phi are not directly accessible before 11.0 but are encoded in the
    // symmetry axis
    auto const [theta, phi] = calculate_theta_phi(solid.GetSymAxis());
    double const alpha_1 = std::atan(solid.GetTanAlpha1());
    double const alpha_2 = std::atan(solid.GetTanAlpha2());
#endif

    return GeoManager::MakeInstance<UnplacedTrapezoid>(
        this->convert_scale_(solid.GetZHalfLength()),
        theta,
        phi,
        this->convert_scale_(solid.GetYHalfLength1()),
        this->convert_scale_(solid.GetXHalfLength1()),
        this->convert_scale_(solid.GetXHalfLength2()),
        alpha_1,
        this->convert_scale_(solid.GetYHalfLength2()),
        this->convert_scale_(solid.GetXHalfLength3()),
        this->convert_scale_(solid.GetXHalfLength4()),
        alpha_2);
}

//---------------------------------------------------------------------------//
//! Convert a simple trapezoid
auto SolidConverter::trd(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Trd const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedTrd>(
        this->convert_scale_(solid.GetXHalfLength1()),
        this->convert_scale_(solid.GetXHalfLength2()),
        this->convert_scale_(solid.GetYHalfLength1()),
        this->convert_scale_(solid.GetYHalfLength2()),
        this->convert_scale_(solid.GetZHalfLength()));
}

//---------------------------------------------------------------------------//
//! Convert a tube section
auto SolidConverter::tubs(arg_type solid_base) -> result_type
{
    auto const& solid = dynamic_cast<G4Tubs const&>(solid_base);
    return GeoManager::MakeInstance<UnplacedTube>(
        this->convert_scale_(solid.GetInnerRadius()),
        this->convert_scale_(solid.GetOuterRadius()),
        this->convert_scale_(solid.GetZHalfLength()),
        solid.GetStartPhiAngle(),
        solid.GetDeltaPhiAngle());
}

//---------------------------------------------------------------------------//
//! Convert a union solid
auto SolidConverter::unionsolid(arg_type solid_base) -> result_type
{
    PlacedBoolVolumes pv = this->convert_bool_impl(
        static_cast<G4BooleanSolid const&>(solid_base));
    return GeoManager::MakeInstance<UnplacedBooleanVolume<kUnion>>(
        kSubtraction, pv[0], pv[1]);
}

//---------------------------------------------------------------------------//
// HELPERS
//---------------------------------------------------------------------------//
//! Create daughter volumes for a boolean solid
auto SolidConverter::convert_bool_impl(G4BooleanSolid const& bs)
    -> PlacedBoolVolumes
{
    Array<G4VSolid const*, 2> solids;
    Array<std::string, 2> labels;
    static Array<char const*, 2> const lr = {{"left", "right"}};
    for (auto i : range(solids.size()))
    {
        solids[i] = bs.GetConstituentSolid(i);
        CELER_ASSERT(solids[i]);

        // Construct name
        std::ostringstream os;
        os << "[TEMP]@" << bs.GetName() << '/' << lr[i] << '/'
           << solids[i]->GetName();
        labels[i] = os.str();
    }

    // The "right" shape should be a G4DisplacedSolid which holds the
    // matrix
    CELER_EXPECT(!dynamic_cast<G4DisplacedSolid const*>(solids[0]));
    CELER_EXPECT(dynamic_cast<G4DisplacedSolid const*>(solids[1]));
    auto const* displaced = static_cast<G4DisplacedSolid const*>(solids[1]);
    solids[1] = displaced->GetConstituentMovedSolid();

    Transformation3D inverse
        = convert_transform_(displaced->GetTransform().Invert());
    Array<Transformation3D const*, 2> transforms
        = {{&Transformation3D::kIdentity, &inverse}};
    Array<VUnplacedVolume const*, 2> converted;
    PlacedBoolVolumes result;

    for (auto i : range(solids.size()))
    {
        converted[i] = (*this)(*solids[i]);
        // XXX: assume VecGeom takes ownership of the LV?
        auto* temp_lv = new LogicalVolume(labels[i].c_str(), converted[i]);
        result[i] = temp_lv->Place(transforms[i]);
    }

    CELER_ENSURE(result[0] && result[1]);
    return result;
}

//---------------------------------------------------------------------------//
//! Compare volumes
void SolidConverter::compare_volumes(G4VSolid const& g4,
                                     vecgeom::VUnplacedVolume const& vg)
{
    if (dynamic_cast<G4BooleanSolid const*>(&g4))
    {
        // Skip comparison of boolean solids because volumes are stochastic
        return;
    }

    auto g4_cap = this->calc_capacity(g4);
    auto vg_cap = vg.Capacity();

    if (CELER_UNLIKELY(!SoftEqual{0.01}(vg_cap, g4_cap)))
    {
        CELER_LOG(warning)
            << "Solid type '" << g4.GetEntityType()
            << "' conversion may have failed: VecGeom/G4 volume ratio is "
            << vg_cap << " / " << g4_cap << " [cm^3] = " << vg_cap / g4_cap;
    }
}

//---------------------------------------------------------------------------//
//! Calculate the capacity in native celeritas units
double SolidConverter::calc_capacity(G4VSolid const& g4) const
{
    return const_cast<G4VSolid&>(g4).GetCubicVolume()
           * ipow<3>(this->convert_scale_(1.0));
}

//---------------------------------------------------------------------------//
}  // namespace g4vg
}  // namespace celeritas
