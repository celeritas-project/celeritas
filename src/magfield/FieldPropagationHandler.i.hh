//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagationHandler.i.hh
//---------------------------------------------------------------------------//

#include "magfield/FieldPropagationHandler.hh"
#include "magfield/ConstFieldHelixStepper.hh"
//#include "Geant/geometry/NavigationInterface.hh"
#include "geometry/GeoTrackView.hh"
#include "base/ArrayUtils.hh"

//#include "Geant/core/VectorTypes.hpp"
//#include "Geant/core/SystemOfUnits.hpp"
//#include "Geant/WorkspaceForFieldPropagation.h"

///#include "VecGeom/navigation/NavigationState.h"

// #define CHECK_VS_RK   1
// #define CHECK_VS_HELIX 1

#define REPORT_AND_CHECK 1

//#define STATS_METHODS 1
// #define DEBUG_FIELD 1

#ifdef CHECK_VS_HELIX
#  define CHECK_VS_SCALAR 1
#endif

#ifdef CHECK_VS_RK
#  define CHECK_VS_SCALAR 1
#endif

namespace celeritas {

constexpr double gEpsDeflection = 1.E-2 * units::cm;
// constexpr auto stageAfterCrossing = SimulationStage::PostPropagationStage;

static constexpr double kB2C = -0.299792458e-3;

// #ifdef STATS_METHODS
// static std::atomic<unsigned long> numTot, numRK, numHelixZ, numHelixGen, numVecRK;
// const unsigned long gPrintStatsMod = 500000;
// #endif

//______________________________________________________________________________
// Curvature for general field
VECCORE_ATT_HOST_DEVICE
double FieldPropagationHandler::Curvature(const GeoTrackView &track) const
{
  ThreeVector_t magFld;
  double bmag = 0.0;

  FieldLookup::GetFieldValue(track.pos(), magFld, bmag);

  return Curvature(track, magFld, bmag);
}

// Needing a real implementation.
double Charge(const GeoTrackView &track)
{
  return double(track.charge());
}

//______________________________________________________________________________
// Curvature for general field
VECCORE_ATT_HOST_DEVICE
double FieldPropagationHandler::Curvature(const GeoTrackView &track,
					    const ThreeVector_t &Bfield,
					    double bmag) const
{
  assert(bmag > 0.0);
  const double& pmag = track.momentum();
  double plong = pmag * dot_product( track.dir(), Bfield) / bmag;
  double pt    = sqrt(pmag * pmag - plong * plong);
  return celeritas::constants::cLight * bmag / pt; // if bmag and pt are positive, no need to call fabs()
}

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
bool FieldPropagationHandler::Propagate(GeoTrackView &track) const
{
  // Scalar geometry length computation. The track is moved into the output basket.
  using vecCore::math::Max;
  using vecCore::math::Min;
  constexpr double step_push = 1.e-4;
  // The minimum step is step_push in case the physics step limit is not smaller
  std::cout<<" FieldPropH: Propagate(): spot 1: pos="<< track <<"\n";
  double step_min = Min(track.pstep(), step_push);
  // The track snext value is already the minimum between geometry and physics
  double step_geom_phys = Max(step_min, track.snext());
  // Field step limit. We use the track sagitta to estimate the "bending" error,
  // i.e. what is the propagated length for which the track deviation in
  // magnetic field with respect to straight propagation is less than epsilon.
  double bmag = -1.0;
  ThreeVector_t BfieldInitial;
  ThreeVector_t Position(track.pos());
  std::cout<<" FieldPropH: Propagate(): spot 2: pos="<< track <<"\n";
  FieldLookup::GetFieldValue(Position, BfieldInitial, bmag);
  double step_field = Max(SafeLength(track, gEpsDeflection, BfieldInitial, bmag),
                          track.safety());

  double step = Min(step_geom_phys, step_field);

  // Propagate in magnetic field
  std::cout<<" FieldPropH: Propagate(): spot 3: pos="<< track <<"\n";
  PropagateInVolume(track, step, BfieldInitial, bmag);
  std::cout<<" FieldPropH: Propagate(): spot 4: pos="<< track <<"\n";
  // Update number of partial steps propagated in field
  // td->fNmag++;

#ifndef IMPLEMENTED_STATUS
// Code moved one level up
// #  warning "Propagation has no way to tell scheduler what to do yet."
#else
  // Set continuous processes stage as follow-up for tracks that reached the
  // physics process
  if (track.status() == SimulationStage::Physics) {
    // Update number of steps to physics and total number of steps
    // td->fNphys++;
    // td->fNsteps++;
    std::cout<<" FieldPropH: Propagate(): spot 5: pos="<< track <<"\n";
    track->SetStage(stageAfterCrossing); // Future: (kPostPropagationStage);
    std::cout<<" FieldPropH: Propagate(): spot 5a: pos="<< track <<"\n";
  } else {
    // Crossing tracks continue to continuous processes, the rest have to
    // query again the geometry
    std::cout<<" FieldPropH: Propagate(): spot 6: pos="<< track <<"\n";
    if ((track->GetSafety() < 1.E-10) && !IsSameLocation(*tracks)) {
      std::cout<<" FieldPropH: Propagate(): spot 6a: pos="<< track <<"\n";
      // td->fNcross++;
      // td->fNsteps++;
    } else {
      std::cout<<" FieldPropH: Propagate(): spot 6b: pos="<< track <<"\n";
      track->SetStage(SimulationStage::GeometryStepStage);
      std::cout<<" FieldPropH: Propagate(): spot 6c: pos="<< track <<"\n";
    }
  }
#endif

  std::cout<<" FieldPropH: Propagate(): spot 7: pos="<< track <<"\n";
  return true;
}

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
void FieldPropagationHandler::PropagateInVolume(GeoTrackView &track, double crtstep,
                                                const ThreeVector_t &BfieldInitial,
                                                double bmag) const
{
  // Single track propagation in a volume. The method is to be called
  // only with  charged tracks in magnetic field.The method decreases the fPstepV
  // fSafetyV and fSnextV with the propagated values while increasing the fStepV.
  // The status and boundary flags are set according to which gets hit first:
  // - physics step (bdr=0)
  // - safety step (bdr=0)
  // - snext step (bdr=1)

  // std::cout << "FieldPropagationHandler::PropagateInVolume called for 1 track" <<
  // std::endl;

  std::cout<<" FieldPropH: Propagate(): spot 1: pos="<< track <<"\n";
  constexpr double toKiloGauss = 1.0 / units::kilogauss; // Converts to kilogauss

// #if ENABLE_MORE_COMPLEX_FIELD
//   bool useRungeKutta   = fPropagator->fConfig->fUseRungeKutta;
//   auto fieldConfig     = FieldLookup::GetFieldConfig();
//   auto fieldPropagator = fFieldPropagator;
//   std::cout<<" FieldPropH: Propagate(): spot 1a: pos="<< track <<"\n";
// #endif

  /****
  #ifndef VECCORE_CUDA_DEVICE_COMPILATION
    auto fieldPropagator = GetFieldPropagator(td);
    if (!fieldPropagator && !td->fSpace4FieldProp) {
      fieldPropagator = Initialize(td);
    }
  #endif
  *****/

#if DEBUG_FIELD
  bool verboseDiff = true; // If false, print just one line.  Else more details.
  bool epsilonRK   = td->fPropagator->fConfig->fEpsilonRK;
#endif

#define PRINT_STEP_SINGLE 1
#ifdef PRINT_STEP_SINGLE
  //GL: denominator should be track Pt w.r.t. Bfield, not P().  Did not fix, as it is only used for debugging printout.
  double curvaturePlus = fabs(kB2C * Charge(track) * (bmag * toKiloGauss)) / (track.momentum() + 1.0e-30); // norm for step
  const double angle = crtstep * curvaturePlus;
  std::cout<<"__PropagateInVolume(Single): Momentum= "<< track.momentum() <<"/"<< units::GeV <<"="<< (track.momentum()/units::GeV)
	   <<" (GeV) Curvature= "<< Curvature(track) * units::mm <<" (1/mm)"
	   <<"; step= "<< crtstep <<"/"<< units::mm <<"="<< crtstep / units::mm <<" (mm), Bmag="<< bmag <<"*"<< toKiloGauss
	   <<" = "<< bmag * toKiloGauss <<" KG   angle= "<< angle << std::endl;
// Print("\n");
#endif

  ThreeVector_t Position(track.pos());
  ThreeVector_t Direction(track.dir());
  ThreeVector_t PositionNew = {0., 0., 0.};
  ThreeVector_t DirectionNew = {0., 0., 0.};

  // char method= '0';

  std::cout<<" FieldPropH: Propagate(): spot 2: pos="<< track <<"\n";
// #if ENABLE_MORE_COMPLEX_FIELD
//   ThreeVector_t PositionNewCheck = {0., 0., 0.};
//   ThreeVector_t DirectionNewCheck = {0., 0., 0.};
//   if (useRungeKutta || !fieldConfig->IsFieldUniform()) {
//     assert(fieldPropagator);
//     std::cout<<" FieldPropH: Propagate(): spot 3: pos="<< track <<"\n";
//     fieldPropagator->DoStep(Position, Direction, Charge(track), track.momentum(), crtstep,
//                             PositionNew, DirectionNew);
//     std::cout<<" FieldPropH: Propagate(): spot 4: pos="<< track <<"\n";
//     assert((PositionNew - Position).Mag() < crtstep + 1.e-4);
// #  ifdef DEBUG_FIELD
// // cross check
// #    ifndef CHECK_VS_BZ
//     ConstFieldHelixStepper stepper(BfieldInitial * toKiloGauss);
//     std::cout<<" FieldPropH: Propagate(): spot 5a: pos="<< track <<"\n";
//     stepper.DoStep<double>(Position, Direction, Charge(track), track.momentum(), crtstep,
//                            PositionNewCheck, DirectionNewCheck);
// #    else
//     double Bz = BfieldInitial[2] * toKiloGauss;
//     ConstBzFieldHelixStepper stepper_bz(Bz); //
//     stepper_bz.DoStep<ThreeVector, double, int>(Position, Direction, Charge(track),
//                                                 track.momentum(), crtstep, PositionNewCheck,
//                                                 DirectionNewCheck);
// #    endif

//     std::cout<<" FieldPropH: Propagate(): spot 6: pos="<< track <<"\n";
//     double posShift = (PositionNew - PositionNewCheck).Mag();
//     double dirShift = (DirectionNew - DirectionNewCheck).Mag();

//     if (posShift > epsilonRK || dirShift > epsilonRK) {
//       std::cout << "*** position/direction shift RK vs. HelixConstBz :" << posShift
//                 << " / " << dirShift << "\n";
//       if (verboseDiff) {
//         printf("%s End> Pos= %9.6f %9.6f %9.6f  Mom= %9.6f %9.6f %9.6f\n",
//                " FPH::PiV(1)-RK: ", PositionNew[0], PositionNew[1], PositionNew[2],
//                DirectionNew[0], DirectionNew[1], DirectionNew[2]);
//         printf("%s End> Pos= %9.6f %9.6f %9.6f  Mom= %9.6f %9.6f %9.6f\n",
//                " FPH::PiV(1)-Bz: ", PositionNewCheck[0], PositionNewCheck[1],
//                PositionNewCheck[2], DirectionNewCheck[0], DirectionNewCheck[1],
//                DirectionNewCheck[2]);
//       }
//     }
// #  endif
// // method= 'R';
// #  ifdef STATS_METHODS
//     numRK++;
//     numTot++;
// #  endif
//   } else {
// #endif

    // geant::
    double BfieldArr[3] = {BfieldInitial[0] * toKiloGauss,
                           BfieldInitial[1] * toKiloGauss,
                           BfieldInitial[2] * toKiloGauss};
    std::cout<<" FieldPropH: Propagate(): spot 7: pos="<< track <<"\n";
    ConstFieldHelixStepper stepper(BfieldArr);
    std::cout<<" FieldPropH: Propagate(): spot 8: pos="<< track <<"\n";
    stepper.DoStep<double>(Position, Direction, Charge(track),
                           track.momentum(), crtstep, PositionNew,
                           DirectionNew);
// method= 'v';
#ifdef STATS_METHODS
    numHelixGen++;
    numTot++;
#endif
// #if ENABLE_MORE_COMPLEX_FIELD
//   }
// #endif

  std::cout<<" FieldPropH: Propagate(): spot 9: pos="<< track <<", newPos="<< PositionNew <<", newDir="<< DirectionNew <<"\n";
#ifdef PRINT_FIELD
  // Print(" FPH::PiV(1): Start>", " Pos= %8.5f %8.5f %8.5f  Mom= %8.5f %8.5f %8.5f",
  // Position[0], Position[1], Position[2], Direction[0], Direction[1], Direction[2]
  // ); Print(" FPH::PiV(1): End>  ", " Pos= %8.5f %8.5f %8.5f  Mom= %8.5f %8.5f %8.5f",
  // PositionNew[0], PositionNew[1], PositionNew[2], DirectionNew[0],
  // DirectionNew[1], DirectionNew[2] );

  // printf(" FPH::PiV(1): ");
  printf(" FPH::PiV(1):: ev= %3d trk= %3d %3d %c ", track.Event(), track.Particle(),
         track.GetNsteps(), method);
  printf("Start> Pos= %8.5f %8.5f %8.5f  Mom= %8.5f %8.5f %8.5f ", Position[0],
         Position[1], Position[2], Direction[0], Direction[1], Direction[2]);
  printf(" s= %10.6f ang= %7.5f ", crtstep / units::mm, angle);
  printf( // " FPH::PiV(1): "
      "End> Pos= %9.6f %9.6f %9.6f  Mom= %9.6f %9.6f %9.6f\n", PositionNew[0],
      PositionNew[1], PositionNew[2], DirectionNew[0], DirectionNew[1],
      DirectionNew[2]);
#endif

#ifdef STATS_METHODS
  unsigned long ntot = numTot;
  if (ntot % gPrintStatsMod < 1) {
    PrintStats();
    // if (numTot > 10 * gPrintStatsMod) gPrintStatsMod = 10 * gPrintStatsMod;
  }
#endif

  //  may normalize direction here  // vecCore::math::Normalize(dirnew);
  normalize_direction(&DirectionNew);
  axpy(-1.0, PositionNew, &Position);
  double posShiftSq = dot_product(Position, Position);

#ifdef COMPLETE_FUNCTIONAL_UPDATES
  std::cout<<" FieldPropH: Propagate(): spot 10: pos="<< track <<"\n";
  track.SetPosition(PositionNew);
  track.SetDirection(DirectionNew);
  track.NormalizeFast();

  // Reset relevant variables
  std::cout<<" FieldPropH: Propagate(): spot 11: pos="<< track <<"\n";
  track.SetStatus(kInFlight);
  track.IncrementNintSteps();
  track.IncreaseStep(crtstep);

  std::cout<<" FieldPropH: Propagate(): spot 12: pos="<< track <<"\n";
  track.DecreasePstep(crtstep);
  if (track.fPhysicsState.fPstep < 1.E-10) {
    track.SetPstep(0);
    track.SetStatus(kPhysics);
  }
  std::cout<<" FieldPropH: Propagate(): spot 13: pos="<< track <<"\n";
  track.DecreaseSnext(crtstep);
  if (track.GetSnext() < 1.E-10) {
    track.SetSnext(0);
    if (track.Boundary()) track.SetStatus(kBoundary);
  }

  std::cout<<" FieldPropH: Propagate(): spot 14: pos="<< track <<"\n";
  double preSafety = track.GetSafety();
  if (posShiftSq > preSafety * preSafety) {
    track.SetSafety(0);
  } else {
    double posShift = std::sqrt(posShiftSq);
    track.DecreaseSafety(posShift);
    if (track.GetSafety() < 1.E-10) track.SetSafety(0);
  }
#else
  track.setPosition(PositionNew);
  track.setDirection(DirectionNew);
  // TODO: need to add normlization of track.
  // Normalize(track)
  //track.dir().Normalize(); // GL: Not needed, DirectionNew is now normalized.
  std::cout<<" fDir new: ("<< track.dir()[0] <<"; "<< track.dir()[1] <<"; "<< track.dir()[2] <<")\n";

  // track.SetStatus(kInFlight);
  track.step() += crtstep;

  std::cout<<" FieldPropH: Propagate(): spot 15: pos="<< track <<"\n";
  track.pstep() -= crtstep;
  if (track.pstep() < 1.E-10) {
    track.pstep() = 0;
    track.status() = GeoTrackStatus::Physics;
  }

  std::cout<<" FieldPropH: Propagate(): spot 16: pos="<< track <<"\n";
  track.snext() -= crtstep;
  if (track.snext() < 1.E-10) {
    track.snext() = 0;
    if (track.boundary() == Boundary::Yes) track.status() = GeoTrackStatus::Boundary;
  }

  std::cout<<" FieldPropH: Propagate(): spot 17: pos="<< track <<"\n";
  const auto preSafety = track.safety();
  if (posShiftSq > preSafety * preSafety) {
    track.safety() = 0;
  } else {
    double posShift = std::sqrt(posShiftSq);
    track.safety() -= posShift;
    if (track.safety() < 1.E-10) track.safety() = 0;
  }
#endif

  std::cout<<" FieldPropH: Propagate(): spot 18: pos="<< track <<"\n";
#ifdef REPORT_AND_CHECK
  CheckTrack(track, "End of Propagate-In-Volume", 1.0e-5);
#endif
}

/*
//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
bool FieldPropagationHandler::IsSameLocation(GeoTrackView &track) const
{
  // Query geometry if the location has changed for a track
  // Returns number of tracks crossing the boundary (0 or 1)

  if (track.safety() > 1.E-10 && track.snext() > 1.E-10) {
    // Track stays in the same volume
    track.fGeometryState.fBoundary = false;
    return true;
  }

  // It might be advantageous to not create the state each time.
  // vecgeom::NavigationState *tmpstate = td->GetPath();
  //vecgeom::NavigationState *tmpstate = vecgeom::NavigationState::MakeInstance(track.fGeometryState.fPath->GetMaxLevel());
  //bool same = NavigationInterface::IsSameLocation(track, *tmpstate);

  bool same = NavigationInterface::IsSameLocation(track);

  //delete tmpstate;

  if (same) {
    track.fGeometryState.fBoundary = false;
    return true;
  }

  track.fGeometryState.fBoundary = true;
  //track.SetStatus(kBoundary);
  if (track.fGeometryState.fNextpath->IsOutside()) {
    track.status() = TrackStatus::ExitingSetup;
  }

  // if (track.GetStep() < 1.E-8) td->fNsmall++;
  return false;
}
*/


#ifdef REPORT_AND_CHECK
#  define IsNan(x) (!(x > 0 || x <= 0.0))
//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
void FieldPropagationHandler::CheckTrack(GeoTrackView &track, const char *msg,
                                         double epsilon) const
{
  // Ensure that values are 'sensible' - else print msg and track
  if (epsilon <= 0.0 || epsilon > 0.01) {
    epsilon = 1.e-6;
  }

  double x = track.pos()[0], y = track.pos()[1], z = track.pos()[2];
  bool badPosition       = IsNan(x) || IsNan(y) || IsNan(z);
  const double maxRadius = 10000.0; // Should be a property of the geometry
  const double maxRadXY  = 5000.0;  // Should be a property of the geometry

  // const double maxUnitDev =  1.0e-4;  // Deviation from unit of the norm of the
  // direction
  double radiusXy2 = x * x + y * y;
  double radius2   = radiusXy2 + z * z;
  badPosition      = badPosition || (radiusXy2 > maxRadXY * maxRadXY) ||
                (radius2 > maxRadius * maxRadius);

  const double maxUnitDev =
      epsilon; // Use epsilon for max deviation of direction norm from 1.0

  double dx = track.dir()[0], dy = track.dir()[1], dz = track.dir()[2];
  double dirNorm2   = dx * dx + dy * dy + dz * dz;
  bool badDirection = std::fabs(dirNorm2 - 1.0) > maxUnitDev;
  if (badPosition || badDirection) {
    static const char *errMsg[4] = {" All ok - No error. ",
                                    " Bad position.",                 // [1]
                                    " Bad direction.",                // [2]
                                    " Bad direction and position. "}; // [3]
    int iM                       = 0;
    if (badPosition) {
      iM++;
    }
    if (badDirection) {
      iM += 2;
    }
    // if( badDirection ) {
    //   Printf( " Norm^2 direction= %f ,  Norm -1 = %g", dirNorm2, sqrt(dirNorm2)-1.0 );
    // }
    printf("ERROR> Problem with track %p . Issue: %s. Info message: %s -- Mag^2(dir)= "
           "%9.6f Norm-1= %g",
           (void *)&track, errMsg[iM], msg, dirNorm2, sqrt(dirNorm2) - 1.0);
    std::cout<< msg <<": track: "<< track << std::endl;
  }
}
#endif

  /*
VECCORE_ATT_HOST
void FieldPropagationHandler::PrintStats() const
{
#ifdef STATS_METHODS
  unsigned long nTot = numTot;
  unsigned long rk = numRK, hZ = numHelixZ, hGen = numHelixGen, vecRK = numVecRK;
  std::cerr << "Step statistics (field Propagation):  total= " << nTot << " RK = " << rk << " ( vecRK = " << vecRK
            << " ) "
            << "  HelixGen = " << hGen << " Helix-Z = " << hZ << std::endl;
#endif
}
*/

} // namespace celeritas
