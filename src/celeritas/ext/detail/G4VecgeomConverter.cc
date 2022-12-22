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

#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/FlatVoxelManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/HybridManager2.h>
#include <VecGeom/navigation/HybridLevelLocator.h>
#include <VecGeom/navigation/HybridNavigator2.h>
#include <VecGeom/navigation/HybridSafetyEstimator.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/navigation/SimpleABBoxLevelLocator.h>
#include <VecGeom/navigation/SimpleABBoxNavigator.h>
#include <VecGeom/navigation/SimpleABBoxSafetyEstimator.h>
#include <VecGeom/navigation/VoxelLevelLocator.h>
#include <VecGeom/navigation/VoxelSafetyEstimator.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/volumes/PlanarPolygon.h>
#include <VecGeom/volumes/UnplacedAssembly.h>
#include <VecGeom/volumes/UnplacedBooleanVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>
#include <VecGeom/volumes/UnplacedCone.h>
#include <VecGeom/volumes/UnplacedCutTube.h>
#include <VecGeom/volumes/UnplacedExtruded.h>
#include <VecGeom/volumes/UnplacedGenTrap.h>
#include <VecGeom/volumes/UnplacedGenericPolycone.h>
#include <VecGeom/volumes/UnplacedOrb.h>
#include <VecGeom/volumes/UnplacedParaboloid.h>
#include <VecGeom/volumes/UnplacedParallelepiped.h>
#include <VecGeom/volumes/UnplacedPolycone.h>
#include <VecGeom/volumes/UnplacedPolyhedron.h>
#include <VecGeom/volumes/UnplacedSExtruVolume.h>
#include <VecGeom/volumes/UnplacedScaledShape.h>
#include <VecGeom/volumes/UnplacedSphere.h>
#include <VecGeom/volumes/UnplacedTet.h>
#include <VecGeom/volumes/UnplacedTorus2.h>
#include <VecGeom/volumes/UnplacedTrapezoid.h>
#include <VecGeom/volumes/UnplacedTrd.h>
#include <VecGeom/volumes/UnplacedTube.h>

//#include "TGeoManager.h"
#include <G4PVDivision.hh>

#include "G4AffineTransform.hh"
#include "G4BooleanSolid.hh"
#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4CutTubs.hh"
#include "G4DisplacedSolid.hh"
#include "G4ExtrudedSolid.hh"
#include "G4GenericPolycone.hh"
#include "G4IntersectionSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4Orb.hh"
#include "G4PVParameterised.hh"
#include "G4Para.hh"
#include "G4Polycone.hh"
#include "G4Polyhedra.hh"
#include "G4ReflectedSolid.hh"
#include "G4Sphere.hh"
#include "G4SubtractionSolid.hh"
#include "G4Tet.hh"
#include "G4Torus.hh"
#include "G4Transform3D.hh"
#include "G4Trap.hh"
#include "G4Trd.hh"
#include "G4Tubs.hh"
#include "G4UnionSolid.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VisExtent.hh"
// more stuff might be needed

#include <cassert>
#include <iostream>
#include <list>

#include "G4Navigator.hh"
#include "G4PropagatorInField.hh"
#include "G4RunManager.hh"
#include "G4TransportationManager.hh"

using namespace vecgeom;

double TrapParametersGetZ(G4Trap const& t)
{
    const double* start = reinterpret_cast<const double*>(&t);

    // 10 derives from sizeof(G4VSolid) + ... + offset
    auto r = start[11];
    assert(r == t.GetZHalfLength());
    return r;
}

void TrapParametersGetOriginalThetaAndPhi(G4Trap const& t,
                                          double&       theta,
                                          double&       phi)
{
    const double* start = reinterpret_cast<const double*>(&t);
    // std::cout << " - Extra check: Dz = " << t.GetZHalfLength() << "
    // double[8-12]:"
    //    <<' '<< start[10] <<' '<< start[11] << "\n";
    // std::cout << " - Extra check: tan(alpha1) = " << t.GetTanAlpha1()
    //    << " vs double[16] = " << start[16] << " [17]= " << start[17] <<
    //    "\n";
    // std::cout << " - Extra check: tan(alpha2) = " << t.GetTanAlpha2()
    //     << " vs double[20] = " << start[20] << " [21]= " << start[21] <<
    //     "\n";

    assert(t.GetZHalfLength() == start[11]);
    double x_peek = start[11]; // tan(theta)*cos(phi)
    double y_peek = start[12]; // tan(theta)*sin(phi)

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

// #define CHANGED_GEANT4 1
#ifdef CHANGED_GEANT4
    G4double g4theta = t.GetTheta();
    G4double g4phi   = t.GetPhi();
    std::cout << " Trap : name " << t.GetName() << " 'computed' parameters:  "
              << " a) theta = " << theta << " vs g4 " << g4theta
              << " diff = " << theta - g4theta << "\n"
              << " b) phi   = " << phi << " vs g4 " << phi << " diff = "
              << phi - g4phi
              // << " Next values = " << start[12] << " , " << start[13]
              << "\n";
    theta = g4theta; // t.GetTheta();
    phi   = g4phi;   // t.GetPhi();
#endif
}

void InitVecGeomNavigators()
{
    for (auto& lvol : vecgeom::GeoManager::Instance().GetLogicalVolumesMap())
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
        std::cout << "*** Conversion of G4 -> VecGeom finished ("
                  << timer.Elapsed() << " s) ***\n";
    }
    GeoManager::Instance().SetWorld(fWorld);
    timer.Start();
    GeoManager::Instance().CloseGeometry();
    timer.Stop();
    if (fVerbose)
    {
        std::cout << "*** Closing VecGeom geometry finished ("
                  << timer.Elapsed() << " s) ***\n";
    }
    fWorld = GeoManager::Instance().GetWorld();

    //
    // setup navigator; by
    //
    /*
    timer.Start();
    fFastG4VGLookup.InitSize(VPlacedVolume::GetIdCount());
    auto iter = fPlacedVolumeMap.begin();
    for (; iter != fPlacedVolumeMap.end(); ++iter) {
      auto placedvols = iter->first;
      auto g4pl       = iter->second;
      fFastG4VGLookup.Insert(placedvols, g4pl);
    }
    timer.Stop();
    std::cout << "*** Setup of syncing lookup structure finished (" <<
  timer.Elapsed() << " s) ***\n";

    // making the navigator
    auto nav =
  #ifdef REPLACE_FULL_NAVIGATOR
        new TG4VecGeomIncNavigator(fFastG4VGLookup); // new G4Navigator();
  #else
        new G4Navigator();
    nav->SetVoxelNavigation(new TG4VecGeomVoxelNavigation(fFastG4VGLookup));
  #endif

    // hooking the navigator
    G4TransportationManager *trMgr =
  G4TransportationManager::GetTransportationManager(); assert(trMgr);
    trMgr->SetNavigatorForTracking(nav);
    G4FieldManager *fieldMgr =
  trMgr->GetPropagatorInField()->GetCurrentFieldManager(); delete
  trMgr->GetPropagatorInField(); trMgr->SetPropagatorInField(new
  G4PropagatorInField(nav, fieldMgr)); trMgr->ActivateNavigator(nav);
    G4EventManager *evtMgr = G4EventManager::GetEventManager();
    if (evtMgr) {
      evtMgr->GetTrackingManager()->GetSteppingManager()->SetNavigator(nav);
    }
    std::cout << "TG4VecGeomNavigator created and registered to
  G4TransportationManager\n";
    // attaching special VecGeom navigators
    timer.Start();
    InitVecGeomNavigators();
    timer.Stop();
    std::cout << "*** Setup of VecGeom navigators finished (" <<
  timer.Elapsed() << " s) ***\n";
    */
}

void G4VecGeomConverter::ExtractReplicatedTransformations(
    G4PVReplica const&                             replica,
    std::vector<vecgeom::Transformation3D const*>& transf) const
{
    // read out parameters
    EAxis  axis;
    int    nReplicas;
    double width;
    double offset;
    bool   consuming;
    replica.GetReplicationData(axis, nReplicas, width, offset, consuming);
    std::cout << axis << " " << nReplicas << " " << width << " " << offset
              << " " << consuming << "\n";
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
            std::cerr << "UNSUPPORTED REPLICATION\n";
        }
    }
    for (int r = 0; r < nReplicas; ++r)
    {
        const auto translation = (-width * (nReplicas - 1) * 0.5 + r * width)
                                 * direction;
        auto tr = new vecgeom::Transformation3D(
            translation[0], translation[1], translation[2]);
        transf.push_back(tr);
    }
}

std::vector<VPlacedVolume const*> const*
G4VecGeomConverter::Convert(G4VPhysicalVolume const* node)
{
    // WARN POTENTIALLY UNSUPPORTED CASE
    if (dynamic_cast<const G4PVParameterised*>(node))
    {
        std::cout << "WARNING: PARAMETRIZED VOLUME FOUND " << node->GetName()
                  << "\n";
    }
    fReplicaTransformations.clear();
    if (auto replica = dynamic_cast<const G4PVReplica*>(node))
    {
        std::cout << "INFO: REPLICA VOLUME FOUND " << replica->GetName()
                  << "\n";
#ifdef ACTIVATEREPLICATION
        ExtractReplicatedTransformations(*replica, fReplicaTransformations);
#endif
    }
    if (dynamic_cast<const G4PVDivision*>(node))
    {
        std::cout << "WARNING: DIVISION VOLUME FOUND " << node->GetName()
                  << "\n";
    }

    if (fPlacedVolumeMap.Contains(node))
    {
        assert(false); // for the moment unsupported
        return GetPlacedVolume(node);
    }

    // convert node transformation
    const auto transformation
        = Convert(node->GetTranslation(), node->GetRotation());
    if (fReplicaTransformations.size() == 0)
    {
        fReplicaTransformations.emplace_back(transformation);
    }

    auto vgvector = new std::vector<VPlacedVolume const*>;

    const auto     g4logical      = node->GetLogicalVolume();
    LogicalVolume* logical_volume = Convert(g4logical);

    // place (all replicas here) ... if normal we will only have one
    // transformation
    for (auto& transf : fReplicaTransformations)
    {
        const VPlacedVolume* placed_volume
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
        const auto daughter_node = g4logical->GetDaughter(i);
        const auto placedvector  = Convert(daughter_node);
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
        transformation = new Transformation3D(t[0], t[1], t[2]);
    }
    else
    {
        // transformation = new Transformation3D(
        //    t[0], t[1], t[2], rot->xx(), rot->xy(), rot->xz(), rot->yx(),
        //    rot->yy(), rot->yz(), rot->zx(), rot->zy(), rot->zz());
        transformation = new Transformation3D(t[0],
                                              t[1],
                                              t[2],
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
    //  if (!dynamic_cast<UnplacedScaledShape const *>(
    //          logical_volume->GetUnplacedVolume()) &&
    //      !dynamic_cast<G4BooleanSolid const *>(
    //          volume->GetSolid())) {
    //    const auto v1 = logical_volume->GetUnplacedVolume()->Capacity();
    //    const auto v2 = volume->GetSolid()->GetCubicVolume();
    //    std::cerr << "v1 " << v1 << " " << v2 << "\n";
    //
    //    assert(v1 > 0.);
    //    assert(std::abs(v1 - v2) / v1 < 0.05);
    //  }
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

    if (fUnplacedVolumeMap.Contains(shape))
        return const_cast<VUnplacedVolume*>(fUnplacedVolumeMap[shape]);

    // Check whether this is already a VecGeom::VUnplacedVolume
    if (auto existingUnplaced = dynamic_cast<VUnplacedVolume const*>(shape))
        return const_cast<VUnplacedVolume*>(existingUnplaced);
    // This can occur if either:
    //  - VecGeom is configured for all G4 solid types
    //  - selected G4 solid types are replaced by VecGeom (e.g. G4UTubs)

    //
    // THE BOX
    if (auto box = dynamic_cast<G4Box const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedBox>(
            box->GetXHalfLength(), box->GetYHalfLength(), box->GetZHalfLength());
    }

    // THE POLYCONE
    if (auto p = dynamic_cast<G4Polycone const*>(shape))
    {
        auto params = p->GetOriginalParameters();
        // fix dimensions - (requires making a copy of some arrays)
        const int                 NZs = params->Num_z_planes;
        std::unique_ptr<double[]> zs(new double[NZs]);    // double zs[NZs];
        std::unique_ptr<double[]> rmins(new double[NZs]); // double rmins[NZs];
        std::unique_ptr<double[]> rmaxs(new double[NZs]); // double rmaxs[NZs];
        for (int i = 0; i < NZs; ++i)
        {
            zs[i]    = params->Z_values[i];
            rmins[i] = params->Rmin[i];
            rmaxs[i] = params->Rmax[i];
        }
        unplaced_volume
            = GeoManager::MakeInstance<UnplacedPolycone>(params->Start_angle,
                                                         params->Opening_angle,
                                                         NZs,
                                                         zs.get(),
                                                         rmins.get(),
                                                         rmaxs.get());
    }

    // Polyhedra
    if (auto pgon = dynamic_cast<G4Polyhedra const*>(shape))
    {
        auto params = pgon->GetOriginalParameters();
        // G4 has a different radius conventions (than TGeo, gdml, VecGeom)!
        const double convertRad
            = std::cos(0.5 * params->Opening_angle / params->numSide);

        // fix dimensions - (requires making a copy of some arrays)
        const int                 NZs = params->Num_z_planes;
        std::unique_ptr<double[]> zs(new double[NZs]);    // double zs[NZs];
        std::unique_ptr<double[]> rmins(new double[NZs]); // double rmins[NZs];
        std::unique_ptr<double[]> rmaxs(new double[NZs]); // double rmaxs[NZs];
        for (int i = 0; i < NZs; ++i)
        {
            zs[i]    = params->Z_values[i];
            rmins[i] = params->Rmin[i] * convertRad;
            rmaxs[i] = params->Rmax[i] * convertRad;
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

    // THE TUBESEG
    if (auto tube = dynamic_cast<G4Tubs const*>(shape))
    {
        unplaced_volume
            = GeoManager::MakeInstance<UnplacedTube>(tube->GetInnerRadius(),
                                                     tube->GetOuterRadius(),
                                                     tube->GetZHalfLength(),
                                                     tube->GetStartPhiAngle(),
                                                     tube->GetDeltaPhiAngle());
    }

    //
    // THE CONESEG
    if (auto cone = dynamic_cast<G4Cons const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedCone>(
            cone->GetInnerRadiusMinusZ(),
            cone->GetOuterRadiusMinusZ(),
            cone->GetInnerRadiusPlusZ(),
            cone->GetOuterRadiusPlusZ(),
            cone->GetZHalfLength(),
            cone->GetStartPhiAngle(),
            cone->GetDeltaPhiAngle());
    }

    // THE TORUS
    if (auto torus = dynamic_cast<G4Torus const*>(shape))
    {
        unplaced_volume
            = GeoManager::MakeInstance<UnplacedTorus2>(torus->GetRmin(),
                                                       torus->GetRmax(),
                                                       torus->GetRtor(),
                                                       torus->GetSPhi(),
                                                       torus->GetDPhi());
    }

    // TRD
    if (auto trd = dynamic_cast<G4Trd const*>(shape))
    {
        unplaced_volume
            = GeoManager::MakeInstance<UnplacedTrd>(trd->GetXHalfLength1(),
                                                    trd->GetXHalfLength2(),
                                                    trd->GetYHalfLength1(),
                                                    trd->GetYHalfLength2(),
                                                    trd->GetZHalfLength());
    }

    // TRAPEZOID
    if (auto p = dynamic_cast<G4Trap const*>(shape))
    {
        double theta;
        double phi;
        TrapParametersGetOriginalThetaAndPhi(*p, theta, phi);
        // std::cerr << "TRAP " << p->GetName() << " Theta= " << theta
        //     << " Phi= " << phi << "\n";
        unplaced_volume = GeoManager::MakeInstance<UnplacedTrapezoid>(
            TrapParametersGetZ(*p),
            theta,
            phi,
            p->GetYHalfLength1(),
            p->GetXHalfLength1(),
            p->GetXHalfLength2(),
            p->GetTanAlpha1(),
            p->GetYHalfLength2(),
            p->GetXHalfLength3(),
            p->GetXHalfLength4(),
            p->GetTanAlpha2());
    }

    if (auto boolean = dynamic_cast<G4BooleanSolid const*>(shape))
    {
        // the "right" shape should be a G4DisplacedSolid which holds the
        // matrix
        if (dynamic_cast<G4DisplacedSolid const*>(
                boolean->GetConstituentSolid(0)))
        {
            assert(false);
        }
        if (!dynamic_cast<G4DisplacedSolid const*>(
                boolean->GetConstituentSolid(1)))
        {
            assert(false);
        }
        G4VSolid const* left  = boolean->GetConstituentSolid(0);
        G4VSolid const* right = boolean->GetConstituentSolid(1);

        G4VSolid*         rightraw = nullptr;
        G4AffineTransform g4righttrans;

        if (auto displaced = dynamic_cast<G4DisplacedSolid const*>(right))
        {
            rightraw     = displaced->GetConstituentMovedSolid();
            g4righttrans = displaced->GetTransform().Invert();
        }

        // need the matrix;
        Transformation3D const* lefttrans
            = &vecgeom::Transformation3D::kIdentity;
        auto                    rot = g4righttrans.NetRotation();
        Transformation3D const* righttrans
            = Convert(g4righttrans.NetTranslation(), &rot);

        // unplaced shapes
        VUnplacedVolume const* leftunplaced  = Convert(left);
        VUnplacedVolume const* rightunplaced = Convert(rightraw);

        assert(leftunplaced != nullptr);
        assert(rightunplaced != nullptr);

        //    // the problem is that I can only place logical volumes
        VPlacedVolume* const leftplaced
            = (new LogicalVolume("inner_virtual", leftunplaced))
                  ->Place(lefttrans);
        VPlacedVolume* const rightplaced
            = (new LogicalVolume("inner_virtual", rightunplaced))
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
            assert(false);
        }
    }
    if (auto p = dynamic_cast<G4ReflectedSolid const*>(shape))
    {
        G4VSolid* underlyingSolid = p->GetConstituentMovedSolid();
        std::cerr << " Reflected solid found: "
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
            std::cerr << "SIMPLE REFLECTION --> CONVERT TO SCALED SHAPE  \n";
            VUnplacedVolume* referenced_shape
                = Convert(p->GetConstituentMovedSolid());

            // implement in terms of scaled shape first of all
            // we could later modify the node directly?
            unplaced_volume = GeoManager::MakeInstance<UnplacedScaledShape>(
                referenced_shape, t.xx(), t.yy(), t.zz());
        }
        else
        {
            std::cerr << "NONSIMPLE REFLECTION in solid" << shape->GetName()
                      << "\n";
            unplaced_volume = new GenericSolid<G4ReflectedSolid>(p);
        }
#else
        std::cerr << "Reflection G4 solid " << shape->GetName()
                  << " -- wrapping G4 implementation.\n";
        unplaced_volume = new GenericSolid<G4ReflectedSolid>(p);
#endif
    }
//#endif
//  // THE PARABOLOID
//  if (shape->IsA() == TGeoParaboloid::Class()) {
//    TGeoParaboloid const *const p = static_cast<TGeoParaboloid const
//    *>(shape);
//
//    unplaced_volume =
//    GeoManager::MakeInstance<UnplacedParaboloid>(p->GetRlo() * LUnit(),
//    p->GetRhi() * LUnit(),
//                                                                   p->GetDz()
//                                                                   *
//                                                                   LUnit());
//  }
//
#ifdef TRIAL_PARA
    // Doesn't compile with current G4 --> TO BE ACTIVATED LATER ON
    if (auto pp = dynamic_cast<G4Para const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedParallelepiped>(
            pp->GetXHalfLength(),
            pp->GetYHalfLength(),
            pp->GetZHalfLength(),
            std::atan(pp->GetTanAlpha()), // pp->GetOriginalAlpha(),
            pp->GetOriginalTheta(),
            pp->GetOriginalPhi());
    }
#endif
    if (auto orb = dynamic_cast<G4Orb const*>(shape))
    {
        unplaced_volume
            = GeoManager::MakeInstance<UnplacedOrb>(orb->GetRadius());
    }
    if (auto sphr = dynamic_cast<G4Sphere const*>(shape))
    {
        unplaced_volume = GeoManager::MakeInstance<UnplacedSphere>(
            sphr->GetInnerRadius(),
            sphr->GetOuterRadius(),
            sphr->GetStartPhiAngle(),
            sphr->GetDeltaPhiAngle(),
            sphr->GetStartThetaAngle(),
            sphr->GetDeltaThetaAngle());
    }
    if (auto gp = dynamic_cast<G4GenericPolycone const*>(shape))
    {
        // auto params = p->GetOriginalParameters();
        // fix dimensions - (requires making a copy of some arrays)
        const int                 nRZs = gp->GetNumRZCorner();
        std::unique_ptr<double[]> zs(new double[nRZs]); // double zs[nRZs];
        std::unique_ptr<double[]> rs(new double[nRZs]); // double rs[nRZs];
        for (int i = 0; i < nRZs; ++i)
        {
            G4PolyconeSideRZ rzCorner = gp->GetCorner(i);
            zs[i]                     = rzCorner.z;
            rs[i]                     = rzCorner.r;
        }
        unplaced_volume = GeoManager::MakeInstance<UnplacedGenericPolycone>(
            gp->GetStartPhi(),
            gp->GetEndPhi() - gp->GetStartPhi(),
            nRZs,
            zs.get(),
            rs.get());
    }
    // #ifdef TRIAL_TET
    if (auto tet = dynamic_cast<G4Tet const*>(shape))
    {
        G4ThreeVector anchor, p1, p2, p3;
        tet->GetVertices(anchor, p1, p2, p3);
        // Else use std::vector<G4ThreeVector> vertices = tet->GetVertices();
        const Vector3D<Precision> pt0(
            anchor.getX(), anchor.getY(), anchor.getZ());
        const Vector3D<Precision> pt1(p1.getX(), p1.getY(), p1.getZ());
        const Vector3D<Precision> pt2(p2.getX(), p2.getY(), p2.getZ());
        const Vector3D<Precision> pt3(p3.getX(), p3.getY(), p3.getZ());
        unplaced_volume
            = GeoManager::MakeInstance<UnplacedTet>(pt0, pt1, pt2, pt3);
    }
    // #endif

    // THE CUT TUBE
    if (auto ct = dynamic_cast<G4CutTubs const*>(shape))
    {
        G4ThreeVector lowNorm = ct->GetLowNorm();
        G4ThreeVector hiNorm  = ct->GetHighNorm();
        unplaced_volume       = GeoManager::MakeInstance<UnplacedCutTube>(
            ct->GetInnerRadius(),
            ct->GetOuterRadius(),
            ct->GetZHalfLength(),
            ct->GetStartPhiAngle(),
            ct->GetDeltaPhiAngle(),
            Vector3D<Precision>(lowNorm[0], lowNorm[1], lowNorm[2]),
            Vector3D<Precision>(hiNorm[0], hiNorm[1], hiNorm[2]));
        // TODO: consider moving this as a specialization to UnplacedTube
    }

#ifdef CHECK_CAPACITY
    // Check capacity as a 'soft' confirmation that the shape / solid was
    // constructed correctly
    if (unplaced_volume)
    {
        double capacityG4 = const_cast<G4VSolid*>(shape)->GetCubicVolume();
        double capacityVg = unplaced_volume->Capacity();
        if (capacityVg < 0)
        {
            std::cerr << "Warning> Capacity given by VecGeom is negative = "
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
                std::cerr << " ERROR> Capacities of Geant4 solid and VecGeom "
                             "solid DIFFER ";
            }
            else
            {
                std::cerr << " WARNING> Difference in capacities seen ";
            }
            int oldPrec = std::cerr.precision(12);
            std::cerr << " for volume " << shape->GetName() << " of type "
                      << shape->GetEntityType() << " : "
                      << " G4 gives " << capacityG4 << " VG gives "
                      << capacityVg << " a relative difference of "
                      << relativeDiff << "\n";
            std::cerr.precision(oldPrec);
        }
        else
        {
            if (std::fabs(relativeDiff) > 1.0e-6)
            {
                int oldPrec2 = std::cout.precision(12);
                std::cout << "  Info: Check for volume " << shape->GetName()
                          << " of type " << shape->GetEntityType() << " : "
                          << " G4 gives " << capacityG4 << " VG gives "
                          << capacityVg << " a relative difference of "
                          << relativeDiff << "\n";
                std::cout.precision(oldPrec2);
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
        unplaced_volume = new GenericSolid<G4VSolid>(shape);
        std::cout << " capacity = " << unplaced_volume->Capacity() << "\n";
    }

    fUnplacedVolumeMap.Set(shape, unplaced_volume);
    return unplaced_volume;
}

void G4VecGeomConverter::PrintNodeTable() const
{
    //  for (auto iter : fPlacedVolumeMap) {
    //    std::cerr << iter.first << " " << iter.second << "\n";
    //    TGeoNode const *n = iter.second;
    //    n->Print();
    //  }
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
