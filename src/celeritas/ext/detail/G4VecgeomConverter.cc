//----------------------------------*-C++-*----------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file G4VecgeomConverter.cc
 *
 * Original code from G4VecGeomNav package by John Apostolakis et.al.
 *
 * Original source:
 *   https://gitlab.cern.ch/VecGeom/g4vecgeomnav/-/raw/7f5d5ec3258d2b7ffbf717e4bd37a3a07285a65f/src/G4VecGeomConverter.cxx
 */
//---------------------------------------------------------------------------//
#include "G4VecgeomConverter.hh"

#include <G4AffineTransform.hh>
#include <G4BooleanSolid.hh>
#include <G4Box.hh>
#include <G4Cons.hh>
#include <G4CutTubs.hh>
#include <G4DisplacedSolid.hh>
#include <G4Ellipsoid.hh>
#include <G4EllipticalCone.hh>
#include <G4EllipticalTube.hh>
#include <G4GenericPolycone.hh>
#include <G4GenericTrap.hh>
#include <G4Hype.hh>
#include <G4IntersectionSolid.hh>
#include <G4LogicalVolume.hh>
#include <G4Navigator.hh>
#include <G4Orb.hh>
#include <G4PVDivision.hh>
#include <G4PVParameterised.hh>
#include <G4Para.hh>
#include <G4Paraboloid.hh>
#include <G4Polycone.hh>
#include <G4Polyhedra.hh>
#include <G4PropagatorInField.hh>
#include <G4ReflectedSolid.hh>
#include <G4Sphere.hh>
#include <G4SubtractionSolid.hh>
#include <G4Tet.hh>
#include <G4Torus.hh>
#include <G4Transform3D.hh>
#include <G4Trap.hh>
#include <G4Trd.hh>
#include <G4Tubs.hh>
#include <G4UnionSolid.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VisExtent.hh>
#include <VecGeom/base/Stopwatch.h>
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

#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"

using namespace vecgeom;

namespace celeritas
{

static constexpr double scale = 0.1; // G4 mm to VecGeom cm scale

double TrapParametersGetZ(G4Trap const& t)
{
    double const* start = reinterpret_cast<double const*>(&t);

    // 10 derives from sizeof(G4VSolid) + ... + offset
    auto r = start[11];
    assert(r == t.GetZHalfLength());
    return r;
}

void TrapParametersGetOriginalThetaAndPhi(G4Trap const& t,
                                          double&       theta,
                                          double&       phi)
{
    double const* start = reinterpret_cast<double const*>(&t);
    assert(t.GetZHalfLength() == start[11]);
    double x_peek = start[11];
    double y_peek = start[12];

    if (x_peek == 0. && y_peek == 0.)
    {
        theta = 0.;
        phi   = 0.;
    }
    else // try to catch more corner cases + requirement that theta > 0??
    {
        // tan(t) = x / cos(phi) --> y = x / cos(phi) * sin(phi) = x * tan(phi)
        phi   = std::atan2(y_peek, x_peek);
        theta = std::atan2(x_peek, cos(phi));
    }
}

void InitVecGeomNavigators()
{
    for (auto& lvol : GeoManager::Instance().GetLogicalVolumesMap())
    {
        if (lvol.second->GetDaughtersp()->size() < 4)
        {
            lvol.second->SetNavigator(NewSimpleNavigator<>::Instance());
        }
        if (lvol.second->GetDaughtersp()->size() >= 5)
        {
            lvol.second->SetNavigator(SimpleABBoxNavigator<>::Instance());
            lvol.second->SetSafetyEstimator(
                SimpleABBoxSafetyEstimator::Instance());
        }
        if (lvol.second->GetDaughtersp()->size() >= 10)
        {
            lvol.second->SetNavigator(HybridNavigator<>::Instance());
            lvol.second->SetSafetyEstimator(VoxelSafetyEstimator::Instance());
            lvol.second->SetLevelLocator(
                TVoxelLevelLocator<false>::GetInstance());
            HybridManager2::Instance().InitStructure((lvol.second));
            FlatVoxelManager::Instance().InitStructure((lvol.second));
        }

        if (lvol.second->ContainsAssembly())
        {
            lvol.second->SetLevelLocator(
                SimpleAssemblyAwareABBoxLevelLocator::GetInstance());
        }
        else
        {
            if (lvol.second->GetLevelLocator() == nullptr)
            {
                lvol.second->SetLevelLocator(
                    SimpleABBoxLevelLocator::GetInstance());
            }
        }
    }
}

#include "GenericSolid.hh"

void G4VecGeomConverter::ConvertG4Geometry(G4VPhysicalVolume const* worldg4)
{
    Clear();
    GeoManager::Instance().Clear();
    Stopwatch timer;
    timer.Start();
    auto volumes = Convert(worldg4);
    assert(volumes->size() == 1);
    fWorld = (*volumes)[0];
    timer.Stop();
    if (fVerbose)
    {
        CELER_LOG(info) << "*** Conversion of G4 -> VecGeom finished ("
            << timer.Elapsed() << " s) ***";
    }
    GeoManager::Instance().SetWorld(fWorld);
    timer.Start();
    GeoManager::Instance().CloseGeometry();
    timer.Stop();
    if (fVerbose)
    {
        CELER_LOG(info) << "*** Closing VecGeom geometry finished ("
            << timer.Elapsed() << " s) ***";
    }
    fWorld = GeoManager::Instance().GetWorld();
}

void G4VecGeomConverter::ExtractReplicatedTransformations(
    G4PVReplica const&                    replica,
    std::vector<Transformation3D const*>& transf) const
{
    // read out parameters
    EAxis  axis;
    int    nReplicas;
    double width;
    double offset;
    bool   consuming;
    replica.GetReplicationData(axis, nReplicas, width, offset, consuming);
    CELER_LOG(debug) << axis << " " << nReplicas << " " << width << " " << offset
              << " " << consuming;
    assert(offset == 0.);
    // for the moment only replication along x,y,z get translation
    Vector3D<double> direction;
    switch (axis)
    {
        case kXAxis: {
            direction.Set(1, 0, 0);
            break;
        }
        case kYAxis: {
            direction.Set(0, 1, 0);
            break;
        }
        case kZAxis: {
            direction.Set(0, 0, 1);
            break;
        }
        default: {
            CELER_LOG(warning) << "UNSUPPORTED REPLICATION\n";
        }
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
G4VecGeomConverter::Convert(G4VPhysicalVolume const* node)
{
    // WARN POTENTIALLY UNSUPPORTED CASE
    if (dynamic_cast<G4PVParameterised const*>(node))
    {
        CELER_LOG(warning) << "PARAMETRIZED VOLUME FOUND " << node->GetName();
    }
    fReplicaTransformations.clear();
    if (auto replica = dynamic_cast<G4PVReplica const*>(node))
    {
        CELER_LOG(info) << "REPLICA VOLUME FOUND " << replica->GetName();
#ifdef ACTIVATEREPLICATION
        ExtractReplicatedTransformations(*replica, fReplicaTransformations);
#endif
    }
    if (dynamic_cast<G4PVDivision const*>(node))
    {
        CELER_LOG(warning) << "DIVISION VOLUME FOUND " << node->GetName();
    }

    if (fPlacedVolumeMap.Contains(node))
    {
        CELER_ASSERT(false); // for the moment unsupported
        return GetPlacedVolume(node);
    }

    // convert node transformation
    auto const transformation
        = Convert(node->GetTranslation(), node->GetRotation());
    if (fReplicaTransformations.size() == 0)
    {
        fReplicaTransformations.emplace_back(transformation);
    }

    auto vgvector = new std::vector<VPlacedVolume const*>;

    auto const g4logical = node->GetLogicalVolume();
    LogicalVolume* logical_volume = Convert(g4logical);

    // place (all replicas here) ... if normal we will only have one
    // transformation
    for (auto& transf : fReplicaTransformations)
    {
        VPlacedVolume const* placed_volume
            = logical_volume->Place(node->GetName(), transf);
        vgvector->emplace_back(placed_volume);
    }

    int remaining_daughters = 0;
    {
        // All or no daughters should have been placed already
        remaining_daughters = g4logical->GetNoDaughters()
                              - logical_volume->GetDaughters().size();
        assert(remaining_daughters <= 0
               || remaining_daughters == (int)g4logical->GetNoDaughters());
    }

    for (int i = 0; i < remaining_daughters; ++i)
    {
        auto const daughter_node = g4logical->GetDaughter(i);
        auto const placedvector = Convert(daughter_node);
        for (auto placed : *placedvector)
        {
            logical_volume->PlaceDaughter((VPlacedVolume*)placed);
        }
    }

    fPlacedVolumeMap.Set(node, vgvector);
    return vgvector;
}

Transformation3D* G4VecGeomConverter::Convert(G4ThreeVector const&    t,
                                              G4RotationMatrix const* rot)
{
    // if (fTransformationMap.Contains(geomatrix)) return
    // const_cast<Transformation3D *>(fTransformationMap[geomatrix]);
    Transformation3D* transformation;
    if (!rot)
    {
        transformation
            = new Transformation3D(scale * t[0], scale * t[1], scale * t[2]);
    }
    else
    {
        // transformation = new Transformation3D(
        //    t[0], t[1], t[2], rot->xx(), rot->xy(), rot->xz(), rot->yx(),
        //    rot->yy(), rot->yz(), rot->zx(), rot->zy(), rot->zz());
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
    // transformation->FixZeroes();
    // transformation->SetProperties();
    // fTransformationMap.Set(geomatrix, transformation);
    return transformation;
}

LogicalVolume* G4VecGeomConverter::Convert(G4LogicalVolume const* volume)
{
    if (fLogicalVolumeMap.Contains(volume))
        return const_cast<LogicalVolume*>(fLogicalVolumeMap[volume]);

    VUnplacedVolume const* unplaced;
    unplaced = Convert(volume->GetSolid());
    LogicalVolume* const logical_volume
        = new LogicalVolume(volume->GetName().c_str(), unplaced);
    fLogicalVolumeMap.Set(volume, logical_volume);

    // can be used to make a cross check for dimensions and other properties
    // make a cross check using cubic volume property
    if (!dynamic_cast<UnplacedScaledShape const*>(
            logical_volume->GetUnplacedVolume())
        && !dynamic_cast<G4BooleanSolid const*>(volume->GetSolid()))
    {
        auto const v1 = logical_volume->GetUnplacedVolume()->Capacity()
                        / ipow<3>(scale);
        auto const v2 = volume->GetSolid()->GetCubicVolume();

        CELER_ASSERT(v1 > 0.);
        CELER_ASSERT(std::fabs((v1 / v2) - 1.0) < 0.01);
    }
    return logical_volume;
}

// the inverse: here we need both the placed volume and logical volume as input
// they should match
// TGeoVolume *G4VecGeomConverter::Convert(VPlacedVolume const *const
// placed_volume,
//                                         LogicalVolume const *const
//                                         logical_volume)
//{
//  assert(placed_volume->GetLogicalVolume() == logical_volume);
//
//  if (fLogicalVolumeMap.Contains(logical_volume)) return
//  const_cast<TGeoVolume *>(fLogicalVolumeMap[logical_volume]);
//
//  const TGeoShape *root_shape = placed_volume->ConvertToRoot();
//  // Some shapes do not exist in ROOT: we need to protect for that
//  if (!root_shape) return nullptr;
//  TGeoVolume *geovolume = new TGeoVolume(logical_volume->GetLabel().c_str(),
//  /* the name */
//                                         root_shape, 0 /* NO MATERIAL FOR THE
//                                         MOMENT */
//                                         );
//
//  fLogicalVolumeMap.Set(geovolume, logical_volume);
//  return geovolume;
//}

VUnplacedVolume* G4VecGeomConverter::Convert(G4VSolid const* shape)
{
    VUnplacedVolume* unplaced_volume = nullptr;

    if (fUnplacedVolumeMap.Contains(shape)) {
        return const_cast<VUnplacedVolume*>(fUnplacedVolumeMap[shape]);
    }

    // Check whether this is already a vecgeom::VUnplacedVolume
    if (auto existingUnplaced = dynamic_cast<VUnplacedVolume const*>(shape))
        return const_cast<VUnplacedVolume*>(existingUnplaced);
    // This can occur if either:
    //  - VecGeom is configured for all G4 solid types
    //  - selected G4 solid types are replaced by VecGeom (e.g. G4UTubs)

    //
    // Box
    else if (auto box = dynamic_cast<G4Box const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedBox>(
            box->GetXHalfLength() * scale,
            box->GetYHalfLength() * scale,
            box->GetZHalfLength() * scale);
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
        std::unique_ptr<double[]> zs(new double[NZs]);    // double zs[NZs];
        std::unique_ptr<double[]> rmins(new double[NZs]); // double rmins[NZs];
        std::unique_ptr<double[]> rmaxs(new double[NZs]); // double rmaxs[NZs];
        for (int i = 0; i < NZs; ++i)
        {
            zs[i]    = scale * params->Z_values[i];
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
        std::unique_ptr<double[]> zs(new double[NZs]);  // double zs[NZs];
        std::unique_ptr<double[]> rmins(new double[NZs]);  // double
                                                           // rmins[NZs];
        std::unique_ptr<double[]> rmaxs(new double[NZs]);  // double
                                                           // rmaxs[NZs];
        for (int i = 0; i < NZs; ++i)
        {
            zs[i] = scale * params->Z_values[i];
            rmins[i] = scale * params->Rmin[i];
            rmaxs[i] = scale * params->Rmax[i];
        }
        unplaced_volume
            = GeoManager::MakeInstance<UnplacedPolycone>(params->Start_angle,
                                                         params->Opening_angle,
                                                         NZs,
                                                         zs.get(),
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
            std::atan(pp->GetTanAlpha()), // pp->GetOriginalAlpha(),
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
        // auto params = p->GetOriginalParameters();
        // fix dimensions - (requires making a copy of some arrays)
        int const nVtx = gt->GetNofVertices();
        std::unique_ptr<double[]> vx(new double[nVtx]);
        std::unique_ptr<double[]> vy(new double[nVtx]);
        for (int i = 0; i < nVtx; ++i)
        {
            G4TwoVector vtx = gt->GetVertex(i);
            vx[i]           = scale * vtx.x();
            vy[i]           = scale * vtx.y();
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
        std::unique_ptr<Vertex[]> vtx(new Vertex[4]); // 3- or 4-side facets
                                                      // only
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
    }

    // Cut tube
    else if (auto ct = dynamic_cast<G4CutTubs const*>(shape))
    {
        G4ThreeVector lowNorm = ct->GetLowNorm();
        G4ThreeVector hiNorm  = ct->GetHighNorm();
        unplaced_volume       = GeoManager::MakeInstance<UnplacedCutTube>(
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
        G4VSolid const* left  = boolean->GetConstituentSolid(0);
        CELER_EXPECT(!dynamic_cast<G4DisplacedSolid const*>(left));

        G4VSolid const* right = boolean->GetConstituentSolid(1);
        CELER_EXPECT(dynamic_cast<G4DisplacedSolid const*>(right));

        G4VSolid*         rightraw = nullptr;
        G4AffineTransform g4righttrans;

        if (auto displaced = dynamic_cast<G4DisplacedSolid const*>(right))
        {
            rightraw     = displaced->GetConstituentMovedSolid();
            g4righttrans = displaced->GetTransform().Invert();
        }

        // need the matrix
        Transformation3D const* lefttrans = &Transformation3D::kIdentity;
        auto                    rot = g4righttrans.NetRotation();
        Transformation3D const* righttrans
            = Convert(g4righttrans.NetTranslation(), &rot);

        // unplaced shapes
        VUnplacedVolume const* leftunplaced  = Convert(left);
        VUnplacedVolume const* rightunplaced = Convert(rightraw);

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
    else if (auto p = dynamic_cast<G4ReflectedSolid const*>(shape))
    {
        G4VSolid* underlyingSolid = p->GetConstituentMovedSolid();
        CELER_LOG(info) << " Reflected solid found: "
            << " volume: " << shape->GetName()
            << " type = " << shape->GetEntityType()
            << "   -- underlying solid: " << underlyingSolid->GetName()
            << " type = " << underlyingSolid->GetEntityType() << "\n";
// #define USE_VG_SCALE_FOR_REFLECTION 1
#ifdef USE_VG_SCALE_FOR_REFLECTION
        auto t = p->GetDirectTransform3D();
        if (t.getTranslation().mag2() == 0.
            && (t.xx() == -1. || t.yy() == -1. || t.zz() == -1.))
        {
            CELER_LOG(info)
                << "SIMPLE REFLECTION -> CONVERT TO SCALED SHAPE\n";
            VUnplacedVolume* referenced_shape
                = Convert(p->GetConstituentMovedSolid());

            // implement in terms of scaled shape first of all
            // we could later modify the node directly?
            unplaced_volume = GeoManager::MakeInstance<UnplacedScaledShape>(
                referenced_shape, t.xx(), t.yy(), t.zz());
        }
        else
        {
            CLER_LOG(info)
                << "NONSIMPLE REFLECTION in solid" << shape->GetName() << "\n";
            unplaced_volume = new celeritas::GenericSolid<G4ReflectedSolid>(p);
        }
#else
        CELER_LOG(info) << "Reflection G4 solid " << shape->GetName()
            << " -- wrapping G4 implementation.\n";
        unplaced_volume = new celeritas::GenericSolid<G4ReflectedSolid>(p);
#endif
    }

#ifdef CHECK_CAPACITY
    // Check capacity as a 'soft' confirmation that the shape / solid was
    // constructed correctly
    if (unplaced_volume)
    {
        double capacityG4 = const_cast<G4VSolid*>(shape)->GetCubicVolume();
        double capacityVg = unplaced_volume->Capacity() / ipow<3>(scale));
        if (capacityVg < 0)
        {
            CELER_LOG(warning) << "Capacity given by VecGeom is negative = "
                      << capacityVg << " for shape" << shape << " of type "
                      << shape->GetEntityType() << "\n";
            capacityVg *= -1.0;
        }
        double relativeDiff = (capacityVg - capacityG4)
                              / (std::max(capacityVg, capacityG4) + 1.0e-50);
        if (std::fabs(relativeDiff) > 1.0e-3)
        {
            if (std::fabs(relativeDiff) > 0.03)
            {
                CELER_LOG(error)
                    << "Capacities of Geant4 solid and VecGeom solid DIFFER.";
            }
            else
            {
                CELER_LOG(warning) << "Minor difference in capacities detected.";
            }
            // TODO: adapt CELER_LOG's precision as in std::cout
            //int oldPrec = std::cout.precision(12);
            CELER_LOG(info) << "for volume " << shape->GetName() << " of type "
                << shape->GetEntityType() << ": G4 gives " << capacityG4
                << " VG gives "<< capacityVg
                << ", a relative difference of "<< relativeDiff;
            //std::cout.precision(oldPrec);
        }
        else
        {
            if (std::fabs(relativeDiff) > 1.0e-6)
            {
                //int oldPrec2 = std::cout.precision(12);
                CELER_LOG(info) << "Check for volume " << shape->GetName()
                    << " of type " << shape->GetEntityType()
                    << ": G4 gives " << capacityG4
                    << " VG gives " << capacityVg
                    << ", a relative difference of "<< relativeDiff;
                //std::cout.precision(oldPrec2);
            }
        }
    }
#endif

    // New volumes should be implemented here...
    if (!unplaced_volume)
    {
        if (true)
        { // fVerbose) {
            printf("Unsupported shape for G4 solid %s, of type %s\n",
                   shape->GetName().c_str(),
                   shape->GetEntityType().c_str());
        }
        unplaced_volume = new celeritas::GenericSolid<G4VSolid>(shape);
        CELER_LOG(debug)
            << " capacity = " << unplaced_volume->Capacity() / ipow<3>(scale);
    }

    fUnplacedVolumeMap.Set(shape, unplaced_volume);
    return unplaced_volume;
}

void G4VecGeomConverter::Clear()
{
    fPlacedVolumeMap.Clear();
    fUnplacedVolumeMap.Clear();
    fLogicalVolumeMap.Clear();
    if (GeoManager::Instance().GetWorld() == fWorld)
    {
        GeoManager::Instance().SetWorld(nullptr);
    }
}
} // namespace celeritas
