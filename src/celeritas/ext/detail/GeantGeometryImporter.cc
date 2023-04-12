//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
// THE ABOVE TEXT APPLIES TO MODIFICATIONS FROM THE ORIGINAL WORK BELOW:
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
// Original work:
// https://gitlab.cern.ch/VecGeom/g4vecgeomnav/-/raw/7f5d5ec3258d2b7ffbf717e4bd37a3a07285a65f/src/G4VecGeomConverter.cxx
// Original code from G4VecGeomNav package by John Apostolakis et al.
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantGeometryImporter.cc
//---------------------------------------------------------------------------//
#include "GeantGeometryImporter.hh"

#include <G4AffineTransform.hh>
#include <G4BooleanSolid.hh>
#include <G4Box.hh>
#include <G4Cons.hh>
#include <G4CutTubs.hh>
#include <G4DisplacedSolid.hh>
#include <G4Ellipsoid.hh>
#include <G4EllipticalCone.hh>
#include <G4EllipticalTube.hh>
#include <G4GDMLWriteStructure.hh>
#include <G4GenericPolycone.hh>
#include <G4GenericTrap.hh>
#include <G4Hype.hh>
#include <G4IntersectionSolid.hh>
#include <G4LogicalVolume.hh>
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
#include <G4VPhysicalVolume.hh>
#include <G4VisExtent.hh>
#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/FlatVoxelManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/HybridManager2.h>
#include <VecGeom/navigation/HybridNavigator2.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/navigation/SimpleABBoxLevelLocator.h>
#include <VecGeom/navigation/SimpleABBoxNavigator.h>
#include <VecGeom/navigation/SimpleABBoxSafetyEstimator.h>
#include <VecGeom/navigation/VoxelLevelLocator.h>
#include <VecGeom/navigation/VoxelSafetyEstimator.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/volumes/UnplacedAssembly.h>
#include <VecGeom/volumes/UnplacedBooleanVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>
#include <VecGeom/volumes/UnplacedCone.h>
#include <VecGeom/volumes/UnplacedCutTube.h>
#include <VecGeom/volumes/UnplacedEllipsoid.h>
#include <VecGeom/volumes/UnplacedEllipticalCone.h>
#include <VecGeom/volumes/UnplacedEllipticalTube.h>
#include <VecGeom/volumes/UnplacedExtruded.h>
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

#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"

#include "GenericSolid.hh"

using namespace vecgeom;

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
static constexpr double scale = 0.1;  // G4 mm to VecGeom cm scale

//---------------------------------------------------------------------------//
Transformation3D*
make_transformation(G4ThreeVector const& t, G4RotationMatrix const* rot)
{
    Transformation3D* transformation;
    if (!rot)
    {
        transformation
            = new Transformation3D(scale * t[0], scale * t[1], scale * t[2]);
    }
    else
    {
        transformation = new Transformation3D(scale * t[0],
                                              scale * t[1],
                                              scale * t[2],
                                              rot->xx(),
                                              rot->yx(),
                                              rot->zx(),
                                              rot->xy(),
                                              rot->yy(),
                                              rot->zy(),
                                              rot->xz(),
                                              rot->yz(),
                                              rot->zz());
    }
    return transformation;
}
}  // namespace

//---------------------------------------------------------------------------//
VPlacedVolume const&
GeantGeometryImporter::operator()(G4VPhysicalVolume const* g4_world)
{
    VecVPlacedVolume const* top_volumes{nullptr};

    GeoManager::Instance().Clear();
    {
        CELER_LOG(status) << "Converting Geant4 geometry to VecGeom";
        ScopedTimeLog scoped_time;

        top_volumes = this->convert(g4_world);
        CELER_ASSERT(top_volumes->size() == 1);
    }

    {
        CELER_LOG(status) << "Finalizing VecGeom";
        ScopedTimeLog scoped_time;
        GeoManager::Instance().SetWorld((*top_volumes)[0]);
        GeoManager::Instance().CloseGeometry();
    }

    // Reset reference after geometry closing
    VPlacedVolume const* world = GeoManager::Instance().GetWorld();
    CELER_ENSURE(world);
    return *world;
}

void GeantGeometryImporter::extract_replicated_transformations(
    G4PVReplica const& replica,
    std::vector<Transformation3D const*>& transf) const
{
    // read out parameters
    EAxis axis;
    int nReplicas;
    double width;
    double offset;
    bool consuming;
    replica.GetReplicationData(axis, nReplicas, width, offset, consuming);
    CELER_LOG(debug) << axis << " " << nReplicas << " " << width << " "
                     << offset << " " << consuming;
    CELER_ASSERT(offset == 0.);

    // for the moment only replication along x,y,z get translation
    Vector3D<double> direction;
    switch (axis)
    {
        case kXAxis:
            direction.Set(1, 0, 0);
            break;
        case kYAxis:
            direction.Set(0, 1, 0);
            break;
        case kZAxis:
            direction.Set(0, 0, 1);
            break;
        default:
                     CELER_ASSERT_UNREACHABLE();
    }
    for (int r = 0; r < nReplicas; ++r)
    {
        auto const translation = (-width * (nReplicas - 1) * 0.5 + r * width)
                                 * direction;
        auto tr = new Transformation3D(
            translation[0], translation[1], translation[2]);
        transf.push_back(tr);
    }
}

std::vector<VPlacedVolume const*> const*
GeantGeometryImporter::convert(G4VPhysicalVolume const* node)
{
    // Warn about potentially unsupported cases
    if (dynamic_cast<G4PVParameterised const*>(node)
        || dynamic_cast<G4PVReplica const*>(node)
        || dynamic_cast<G4PVDivision const*>(node))
    {
        TypeDemangler<G4VPhysicalVolume> demangle_type;
        CELER_LOG(warning) << "Encountered possibly unsupported Geant4 physical volume type '"
            << demangle_type(*node) << "'";
    }

    // convert node transformation
    auto* transformation
        = make_transformation(node->GetTranslation(), node->GetRotation());
    replica_transformations_.clear();
    replica_transformations_.push_back(transformation);

    std::vector<VPlacedVolume const*> vgvector;

    auto* const g4logical = node->GetLogicalVolume();
    LogicalVolume* logical_volume = this->convert(g4logical);

    // place (all replicas here) ... if normal we will only have one
    // transformation
    for (auto& transf : replica_transformations_)
    {
        VPlacedVolume const* placed_volume
            = logical_volume->Place(node->GetName(), transf);
        vgvector.push_back(placed_volume);
    }

    // All or no daughters should have been placed already
    int remaining_daughters = g4logical->GetNoDaughters()
                                - logical_volume->GetDaughters().size();
    CELER_ASSERT(remaining_daughters <= 0
                 || remaining_daughters == (int)g4logical->GetNoDaughters());

    for (int i = 0; i < remaining_daughters; ++i)
    {
        auto* const daughter_node = g4logical->GetDaughter(i);
        auto* const placedvector = this->convert(daughter_node);
        for (auto* placed : *placedvector)
        {
                     logical_volume->PlaceDaughter(
                         const_cast<VPlacedVolume*>(placed));
        }
    }

    // Move to map, return reference to stored vector
    auto&& [iter, inserted]
        = placed_volume_map_.insert({node, std::move(vgvector)});
    CELER_ASSERT(inserted);
    return &iter->second;
}

LogicalVolume* GeantGeometryImporter::convert(G4LogicalVolume const* g4_logvol)
{
    if (logical_volume_map_.find(g4_logvol) != logical_volume_map_.end())
        return const_cast<LogicalVolume*>(logical_volume_map_[g4_logvol]);

    VUnplacedVolume const* unplaced;
    unplaced = this->convert(g4_logvol->GetSolid());

    // add 0x suffix, unless already provided from GDML through Geant4 parser
    static G4GDMLWriteStructure gdml_mangler;
    std::string clean_name(g4_logvol->GetName());  // may have suffix from GDML
    if (clean_name.find("0x") == std::string::npos)
    {
        // but if not found, add the 0x suffix here
        clean_name = gdml_mangler.GenerateName(clean_name.c_str(), g4_logvol);
    }

    LogicalVolume* const vg_logvol
        = new LogicalVolume(clean_name.c_str(), unplaced);
    logical_volume_map_[g4_logvol] = vg_logvol;

    // can be used to make a cross check for dimensions and other properties
    // make a cross check using cubic volume property
    if (!dynamic_cast<UnplacedScaledShape const*>(vg_logvol->GetUnplacedVolume())
        && !dynamic_cast<G4BooleanSolid const*>(g4_logvol->GetSolid()))
    {
        auto v1 = vg_logvol->GetUnplacedVolume()->Capacity() / ipow<3>(scale);
        auto v2 = g4_logvol->GetSolid()->GetCubicVolume();

        CELER_ASSERT(v1 > 0.);
        CELER_ASSERT(SoftEqual{0.01}(v1, v2));
    }
    return vg_logvol;
}

VUnplacedVolume* GeantGeometryImporter::convert(G4VSolid const* shape)
{
    VUnplacedVolume* unplaced_volume = nullptr;

    if (unplaced_volume_map_.find(shape) != unplaced_volume_map_.end())
    {
        return const_cast<VUnplacedVolume*>(unplaced_volume_map_[shape]);
    }

    // Check whether this is already a vecgeom::VUnplacedVolume
    if (auto existing_unplaced = dynamic_cast<VUnplacedVolume const*>(shape))
        return const_cast<VUnplacedVolume*>(existing_unplaced);
    // This can occur if either:
    //  - VecGeom is configured for all G4 solid types
    //  - selected G4 solid types are replaced by VecGeom (e.g. G4UTubs)

    // Box
    else if (auto box = dynamic_cast<G4Box const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedBox>(
            scale * box->GetXHalfLength(),
            scale * box->GetYHalfLength(),
            scale * box->GetZHalfLength());
    }

    // TRD
    else if (auto trd = dynamic_cast<G4Trd const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedTrd>(
            scale * trd->GetXHalfLength1(),
            scale * trd->GetXHalfLength2(),
            scale * trd->GetYHalfLength1(),
            scale * trd->GetYHalfLength2(),
            scale * trd->GetZHalfLength());
    }

    // Trapezoid
    else if (auto p = dynamic_cast<G4Trap const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedTrapezoid>(
            scale * p->GetZHalfLength(),
            p->GetTheta(),
            p->GetPhi(),
            scale * p->GetYHalfLength1(),
            scale * p->GetXHalfLength1(),
            scale * p->GetXHalfLength2(),
            p->GetAlpha1(),
            scale * p->GetYHalfLength2(),
            scale * p->GetXHalfLength3(),
            scale * p->GetXHalfLength4(),
            p->GetAlpha2());
    }

    // Tube section
    else if (auto tube = dynamic_cast<G4Tubs const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedTube>(
            scale * tube->GetInnerRadius(),
            scale * tube->GetOuterRadius(),
            scale * tube->GetZHalfLength(),
            tube->GetStartPhiAngle(),
            tube->GetDeltaPhiAngle());
    }

    // Cone section
    else if (auto cone = dynamic_cast<G4Cons const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedCone>(
            scale * cone->GetInnerRadiusMinusZ(),
            scale * cone->GetOuterRadiusMinusZ(),
            scale * cone->GetInnerRadiusPlusZ(),
            scale * cone->GetOuterRadiusPlusZ(),
            scale * cone->GetZHalfLength(),
            cone->GetStartPhiAngle(),
            cone->GetDeltaPhiAngle());
    }

    // Polyhedra
    else if (auto pgon = dynamic_cast<G4Polyhedra const*>(shape))
    {
        auto params = pgon->GetOriginalParameters();
        // G4 has a different radius conventions (than TGeo, gdml, VecGeom)!
        double const convertRad
            = std::cos(0.5 * params->Opening_angle / params->numSide);

        // fix dimensions - (requires making a copy of some arrays)
        int const NZs = params->Num_z_planes;
        std::unique_ptr<double[]> zs(new double[NZs]);
        std::unique_ptr<double[]> rmins(new double[NZs]);
        std::unique_ptr<double[]> rmaxs(new double[NZs]);
        for (int i = 0; i < NZs; ++i)
        {
            zs[i] = scale * params->Z_values[i];
            rmins[i] = scale * params->Rmin[i] * convertRad;
            rmaxs[i] = scale * params->Rmax[i] * convertRad;
        }

        auto phistart = params->Start_angle;
        while (phistart < 0.)
        {
            phistart += M_PI * 2.;
        }

        unplaced_volume = GeoManager::MakeInstance<UnplacedPolyhedron>(
            phistart,
            params->Opening_angle,
            params->numSide,
            NZs,
            zs.get(),
            rmins.get(),
            rmaxs.get());
    }

    // Polycone
    else if (auto p = dynamic_cast<G4Polycone const*>(shape))
    {
        auto params = p->GetOriginalParameters();
        // fix dimensions - (requires making a copy of some arrays)
        int const NZs = params->Num_z_planes;
        std::unique_ptr<double[]> zvals(new double[NZs]);
        std::unique_ptr<double[]> rmins(new double[NZs]);
        std::unique_ptr<double[]> rmaxs(new double[NZs]);
        for (int i = 0; i < NZs; ++i)
        {
            zvals[i] = scale * params->Z_values[i];
            rmins[i] = scale * params->Rmin[i];
            rmaxs[i] = scale * params->Rmax[i];
        }
        unplaced_volume
            = GeoManager::MakeInstance<UnplacedPolycone>(params->Start_angle,
                                                         params->Opening_angle,
                                                         NZs,
                                                         zvals.get(),
                                                         rmins.get(),
                                                         rmaxs.get());
    }

    // Generic polycone
    else if (auto gp = dynamic_cast<G4GenericPolycone const*>(shape))
    {
        // fix dimensions - (requires making a copy of some arrays)
        int const nRZs = gp->GetNumRZCorner();
        std::unique_ptr<double[]> zs(new double[nRZs]);
        std::unique_ptr<double[]> rs(new double[nRZs]);
        for (int i = 0; i < nRZs; ++i)
        {
            G4PolyconeSideRZ rzCorner = gp->GetCorner(i);
            zs[i] = scale * rzCorner.z;
            rs[i] = scale * rzCorner.r;
        }
        unplaced_volume = GeoManager::MakeInstance<UnplacedGenericPolycone>(
            gp->GetStartPhi(),
            gp->GetEndPhi() - gp->GetStartPhi(),
            nRZs,
            rs.get(),
            zs.get());
    }

    // Torus
    else if (auto torus = dynamic_cast<G4Torus const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedTorus2>(
            scale * torus->GetRmin(),
            scale * torus->GetRmax(),
            scale * torus->GetRtor(),
            torus->GetSPhi(),
            torus->GetDPhi());
    }

    // Parallelepiped
    else if (auto pp = dynamic_cast<G4Para const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedParallelepiped>(
            scale * pp->GetXHalfLength(),
            scale * pp->GetYHalfLength(),
            scale * pp->GetZHalfLength(),
            std::atan(pp->GetTanAlpha()),
            pp->GetTheta(),
            pp->GetPhi());
    }

    // Hyperbolae
    else if (auto hype = dynamic_cast<G4Hype const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedHype>(
            scale * hype->GetInnerRadius(),
            scale * hype->GetOuterRadius(),
            hype->GetInnerStereo(),
            hype->GetOuterStereo(),
            scale * hype->GetZHalfLength());
    }

    // Orb
    else if (auto orb = dynamic_cast<G4Orb const*>(shape))
    {
        unplaced_volume
            = GeoManager::MakeInstance<UnplacedOrb>(scale * orb->GetRadius());
    }

    // Sphere
    else if (auto sphr = dynamic_cast<G4Sphere const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedSphere>(
            scale * sphr->GetInnerRadius(),
            scale * sphr->GetOuterRadius(),
            sphr->GetStartPhiAngle(),
            sphr->GetDeltaPhiAngle(),
            sphr->GetStartThetaAngle(),
            sphr->GetDeltaThetaAngle());
    }

    // Ellipsoid
    else if (auto ell = dynamic_cast<G4Ellipsoid const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedEllipsoid>(
            scale * ell->GetDx(),
            scale * ell->GetDy(),
            scale * ell->GetDz(),
            scale * ell->GetZBottomCut(),
            scale * ell->GetZTopCut());
    }

    // Elliptical cone
    else if (auto elc = dynamic_cast<G4EllipticalCone const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedEllipticalCone>(
            elc->GetSemiAxisX(),
            elc->GetSemiAxisY(),
            scale * elc->GetZMax(),
            scale * elc->GetZTopCut());
    }

    // Elliptical tube
    else if (auto elt = dynamic_cast<G4EllipticalTube const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedEllipticalTube>(
            scale * elt->GetDx(), scale * elt->GetDy(), scale * elt->GetDz());
    }

    // Tetrahedron
    else if (auto tet = dynamic_cast<G4Tet const*>(shape))
    {
        G4ThreeVector anchor, p1, p2, p3;
        tet->GetVertices(anchor, p1, p2, p3);
        // Else use std::vector<G4ThreeVector> vertices = tet->GetVertices();
        Vector3D<Precision> const pt0(
            anchor.getX(), anchor.getY(), anchor.getZ());
        Vector3D<Precision> const pt1(p1.getX(), p1.getY(), p1.getZ());
        Vector3D<Precision> const pt2(p2.getX(), p2.getY(), p2.getZ());
        Vector3D<Precision> const pt3(p3.getX(), p3.getY(), p3.getZ());
        unplaced_volume = GeoManager::MakeInstance<UnplacedTet>(
            scale * pt0, scale * pt1, scale * pt2, scale * pt3);
    }

    // Generic trapezoid
    else if (auto gt = dynamic_cast<G4GenericTrap const*>(shape))
    {
        // fix dimensions - (requires making a copy of some arrays)
        int const nVtx = gt->GetNofVertices();
        std::unique_ptr<double[]> vx(new double[nVtx]);
        std::unique_ptr<double[]> vy(new double[nVtx]);
        for (int i = 0; i < nVtx; ++i)
        {
            G4TwoVector vtx = gt->GetVertex(i);
            vx[i] = scale * vtx.x();
            vy[i] = scale * vtx.y();
        }
        unplaced_volume = GeoManager::MakeInstance<UnplacedGenTrap>(
            vx.get(), vy.get(), scale * gt->GetZHalfLength());
    }

    // Tessellated solids
    else if (auto tess = dynamic_cast<G4TessellatedSolid const*>(shape))
    {
        using Vertex = vecgeom::Vector3D<vecgeom::Precision>;

        unplaced_volume = GeoManager::MakeInstance<UnplacedTessellated>();
        UnplacedTessellated* unpvol
            = static_cast<UnplacedTessellated*>(unplaced_volume);

        int const nFacets = tess->GetNumberOfFacets();
        std::unique_ptr<Vertex[]> vtx(new Vertex[4]);  // 3- or 4-side facets
        for (int i = 0; i < nFacets; ++i)
        {
            G4VFacet const* facet = tess->GetFacet(i);
            int const nVtx = facet->GetNumberOfVertices();
            for (int iv = 0; iv < nVtx; ++iv)
            {
                auto vxg4 = facet->GetVertex(iv);
                vtx[iv].Set(
                    scale * vxg4.x(), scale * vxg4.y(), scale * vxg4.z());
            }

            if (nVtx == 3)
            {
                unpvol->AddTriangularFacet(vtx[0], vtx[1], vtx[2], ABSOLUTE);
            }
            else
            {
                unpvol->AddQuadrilateralFacet(
                    vtx[0], vtx[1], vtx[2], vtx[3], ABSOLUTE);
            }
        }
        unpvol->Close();
    }

    // Cut tube
    else if (auto ct = dynamic_cast<G4CutTubs const*>(shape))
    {
        G4ThreeVector lowNorm = ct->GetLowNorm();
        G4ThreeVector hiNorm = ct->GetHighNorm();
        unplaced_volume = GeoManager::MakeInstance<UnplacedCutTube>(
            scale * ct->GetInnerRadius(),
            scale * ct->GetOuterRadius(),
            scale * ct->GetZHalfLength(),
            ct->GetStartPhiAngle(),
            ct->GetDeltaPhiAngle(),
            Vector3D<Precision>(lowNorm[0], lowNorm[1], lowNorm[2]),
            Vector3D<Precision>(hiNorm[0], hiNorm[1], hiNorm[2]));
        // TODO: consider moving this as a specialization to UnplacedTube
    }

    // Paraboloid
    else if (auto pb = dynamic_cast<G4Paraboloid const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedParaboloid>(
            scale * pb->GetRadiusMinusZ(),
            scale * pb->GetRadiusPlusZ(),
            scale * pb->GetZHalfLength());
    }

    // Boolean volumes
    else if (auto boolean = dynamic_cast<G4BooleanSolid const*>(shape))
    {
        // the "right" shape should be a G4DisplacedSolid which holds the
        // matrix
        G4VSolid const* left = boolean->GetConstituentSolid(0);
        CELER_EXPECT(!dynamic_cast<G4DisplacedSolid const*>(left));

        G4VSolid const* right = boolean->GetConstituentSolid(1);
        CELER_EXPECT(dynamic_cast<G4DisplacedSolid const*>(right));

        G4VSolid* rightraw = nullptr;
        G4AffineTransform g4righttrans;

        if (auto displaced = dynamic_cast<G4DisplacedSolid const*>(right))
        {
            rightraw = displaced->GetConstituentMovedSolid();
            g4righttrans = displaced->GetTransform().Invert();
        }

        // need the matrix
        Transformation3D const* lefttrans = &Transformation3D::kIdentity;
        auto rot = g4righttrans.NetRotation();
        Transformation3D const* righttrans
            = make_transformation(g4righttrans.NetTranslation(), &rot);

        // unplaced shapes
        VUnplacedVolume const* leftunplaced = this->convert(left);
        VUnplacedVolume const* rightunplaced = this->convert(rightraw);

        CELER_ASSERT(leftunplaced != nullptr);
        CELER_ASSERT(rightunplaced != nullptr);

        // the problem is that I can only place logical volumes
        std::string left_label{left->GetName() + "_bool_left"};
        std::string right_label{right->GetName() + "_bool_right"};
        VPlacedVolume* const leftplaced
            = (new LogicalVolume(left_label.c_str(), leftunplaced))
                  ->Place(lefttrans);
        VPlacedVolume* const rightplaced
            = (new LogicalVolume(right_label.c_str(), rightunplaced))
                  ->Place(righttrans);

        if (dynamic_cast<G4SubtractionSolid const*>(boolean))
        {
            unplaced_volume
                = GeoManager::MakeInstance<UnplacedBooleanVolume<kSubtraction>>(
                    kSubtraction, leftplaced, rightplaced);
        }
        else if (dynamic_cast<G4IntersectionSolid const*>(boolean))
        {
            unplaced_volume
                = GeoManager::MakeInstance<UnplacedBooleanVolume<kIntersection>>(
                    kIntersection, leftplaced, rightplaced);
        }
        else if (dynamic_cast<G4UnionSolid const*>(boolean))
        {
            unplaced_volume
                = GeoManager::MakeInstance<UnplacedBooleanVolume<kUnion>>(
                    kUnion, leftplaced, rightplaced);
        }
        else
        {
            CELER_ASSERT_UNREACHABLE();
        }
    }
    else if (auto refl = dynamic_cast<G4ReflectedSolid const*>(shape))
    {
        G4VSolid* underlyingSolid = refl->GetConstituentMovedSolid();
        CELER_LOG(info)
            << " Reflected solid found: "
            << " volume: " << refl->GetName()
            << " type = " << refl->GetEntityType()
            << "   -- underlying solid: " << underlyingSolid->GetName()
            << " type = " << underlyingSolid->GetEntityType();
        unplaced_volume = new GenericSolid<G4ReflectedSolid>(refl);
    }

    // New volumes should be implemented here...
    if (!unplaced_volume)
    {
        CELER_LOG(info) << "Unsupported shape for G4 solid "
                        << shape->GetName().c_str() << ", of type "
                        << shape->GetEntityType().c_str();
        unplaced_volume = new GenericSolid<G4VSolid>(shape);
        CELER_LOG(debug) << " -- capacity = "
                         << unplaced_volume->Capacity() / ipow<3>(scale);
    }

    unplaced_volume_map_[shape] = unplaced_volume;
    return unplaced_volume;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
