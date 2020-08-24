//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ConstFieldHelixStepper.hh
//---------------------------------------------------------------------------//
/**
 * @brief Interface between the scheduler and the field integrator.
 *
 *  Started from Const[Bz]FieldHelixStepper by S. Wenzel and J. Apostolakis
 */
//===----------------------------------------------------------------------===//
#pragma once

//#include <VecGeom/base/Vector3D.h>
#include "base/Macros.hh" //<Geant/core/Config.hpp>
#include "base/Assert.hh"
#include "base/ArrayUtils.hh"
//#include <Geant/core/VectorTypes.hpp>
#include <iostream>

namespace celeritas {

/**
 * A very simple stepper treating the propagation of particles in a constant Bz magnetic field
 * ( neglecting energy loss of particle )
 * This class is roughly equivalent to TGeoHelix in ROOT
 */
class ConstFieldHelixStepper {

public:
  CELER_FUNCTION
  ConstFieldHelixStepper(double Bx, double By, double Bz);

  CELER_FUNCTION
  ConstFieldHelixStepper(double Bfield[3]);

  CELER_FUNCTION
  ConstFieldHelixStepper(Real3 const &Bfield);

  void SetB(double Bx, double By, double Bz)
  {
    fB = {Bx, By, Bz};
    CalculateDerived();
  }
  //Real3 const &GetB() const { return fB; }

  /**
   * this function propagates the track along the helix solution by a step
   * input: current position, current direction, some particle properties
   * output: new position, new direction of particle
   */
  // template <typename Real_v>
  // CELER_FORCEINLINE_FUNCTION void DoStep(Real_v const &posx, Real_v const &posy, Real_v const &posz,
  // 					 Real_v const &dirx, Real_v const &diry, Real_v const &dirz,
  // 					 Real_v const &charge, Real_v const &momentum,
  // 					 Real_v const &step, Real_v &newposx, Real_v &newposy,
  // 					 Real_v &newposz, Real_v &newdirx, Real_v &newdiry,
  // 					 Real_v &newdirz) const;

  /**
   * basket version of dostep
   * version that takes plain arrays as input; suited for first-gen GeantV
   */
  // template <typename Real_v>
  // CELER_FORCEINLINE_FUNCTION void DoStepArr(Real_v const *posx, Real_v const *posy, Real_v const *posz, Real_v const *dirx,
  // 					    Real_v const *diry, Real_v const *dirz, Real_v const *charge,
  // 					    Real_v const *momentum, Real_v const *step, Real_v *newposx, Real_v *newposy,
  // 					    Real_v *newposz, Real_v *newdirx, Real_v *newdiry, Real_v *newdirz, int np) const;


  // in future will offer versions that take containers as input

  /**
   * this function propagates the track along the helix solution by a step
   * input: current position, current direction, some particle properties
   * output: new position, new direction of particle
   */
  template <typename Real_v>
  CELER_FORCEINLINE_FUNCTION void DoStep(Real3 const &position, Real3 const &direction,
					 Real_v const &charge, Real_v const &momentum, Real_v const &step,
					 Real3 &newPosition, Real3 &newDirection) const;

  // Auxiliary methods
  template <typename Real_v>
  CELER_FORCEINLINE_FUNCTION void PrintStep(Real3 const &startPosition,
					    Real3 const &startDirection, Real_v const &charge,
					    Real_v const &momentum, Real_v const &step, Real3 &endPosition,
					    Real3 &endDirection) const;

protected:
  void CalculateDerived();

private:
  Real3 fB;
  // Auxilary members - calculated from above - cached for speed, code simplicity
  double fBmag;
  Real3 fUnit;
}; // end class declaration

CELER_FORCEINLINE_FUNCTION
void ConstFieldHelixStepper::CalculateDerived()
{
  fBmag = norm<double>(fB);
  fUnit = fB;
  normalize_direction(&fUnit);
}

CELER_FORCEINLINE_FUNCTION
ConstFieldHelixStepper::ConstFieldHelixStepper(double Bx, double By, double Bz)
{
  fB = {Bx, By, Bz};
  CalculateDerived();
}

CELER_FORCEINLINE_FUNCTION
ConstFieldHelixStepper::ConstFieldHelixStepper(double B[3])
{
  fB = {B[0], B[1], B[2]};
  CalculateDerived();
}

CELER_FORCEINLINE_FUNCTION
ConstFieldHelixStepper::ConstFieldHelixStepper(Real3 const &Bfield) : fB(Bfield)
{
  CalculateDerived();
}

/**
 * this function propagates the track along the "helix-solution" by a distance 'step'
 * input: current position (x0, y0, z0), current direction ( dirX0, dirY0, dirZ0 ), some particle properties
 * output: new position, new direction of particle

template <typename Real_v>
CELER_FORCEINLINE_FUNCTION void ConstFieldHelixStepper::DoStep(Real_v const &x0, Real_v const &y0, Real_v const &z0,
                                                       Real_v const &dirX0, Real_v const &dirY0, Real_v const &dirZ0,
                                                       Real_v const &charge, Real_v const &momentum, Real_v const &step,
                                                       Real_v &x, Real_v &y, Real_v &z, Real_v &dx, Real_v &dy,
                                                       Real_v &dz) const
{
  Real3 startPosition(x0, y0, z0);
  Real3 startDirection(dirX0, dirY0, dirZ0);
  Real3 endPosition, endDirection;

  // startPosition.Set( x0, y0, z0);
  // startDirection.Set( dirX0, dirY0, dirZ0);

  DoStep(startPosition, startDirection, charge, momentum, step, endPosition, endDirection);
  x  = endPosition[0];
  y  = endPosition[1];
  z  = endPosition[2];
  dx = endDirection[0];
  dy = endDirection[1];
  dz = endDirection[2];

  // PrintStep(startPosition, startDirection, charge, momentum, step, endPosition, endDirection);
}
  */

template <typename Real_v>
CELER_FORCEINLINE_FUNCTION void ConstFieldHelixStepper::DoStep(Real3 const &startPosition,
                                                       Real3 const &startDirection,
                                                       Real_v const &charge, Real_v const &momentum, Real_v const &step,
                                                       Real3 &endPosition,
                                                       Real3 &endDirection) const
{
  const Real_v kB2C_local(-0.299792458e-3);
  const Real_v kSmall(1.E-30);
  //using vecCore::math::Max;
  //using vecCore::math::SinCos;
  // using vecgeom::Vector3D;
  // using vecCore::math::Sin;
  // using vecCore::math::Cos;
  //using vecCore::math::Abs;
  //using vecCore::math::Sqrt;
  // could do a fast square root here

  // Real_v dt = Sqrt((dx0*dx0) + (dy0*dy0)) + kSmall;

  // std::cout << " ConstFieldHelixStepper::DoStep called.  fBmag= " << fBmag
  //          << " unit dir= " << fUnit << std::endl;

  // assert( std::abs( startDirection.Mag2() - 1.0 ) < 1.0e-6 );
  std::cout<<" ConstFieldHelixStepper DoStep(): spot 1: pos="<< startPosition <<", momentum="<< momentum <<"\n";

  Real3 dir1Field(fUnit);
  Real_v UVdotUB = dot_product(startDirection, dir1Field); //  Limit cases 0.0 and 1.0
  Real_v dt2     = std::max(dot_product(startDirection, startDirection) - UVdotUB * UVdotUB, 0.0);
  Real_v sinVB   = sqrt(dt2) + kSmall;

  // Real_v invnorm = 1. / sinVB;

  // radius has sign and determines the sense of rotation
  Real_v R = momentum * sinVB / (kB2C_local * charge * fBmag);

  Real3 dirVelX = startDirection;
  axpy(-UVdotUB, dir1Field, &dirVelX);
  normalize_direction( &dirVelX );
  Real3 dirCrossVB = cross_product(dirVelX, dir1Field); // OK if it is zero
  // Real3 dirCrossVB = restVelX.Cross(dir1Field);  // OK if it is zero

  /***
  printf("\n");
  printf("CVFHS> dir-1  B-fld  = %f %f %f   mag-1= %g \n", dir1Field[0], dir1Field[1], dir1Field[2],
         dir1Field.Mag()-1.0 );
  printf("CVFHS> dir-2  VelX   = %f %f %f   mag-1= %g \n", dirVelX[0], dirVelX[1], dirVelX[2],
         dirVelX.Mag()-1.0 );
  printf("CVFHS> dir-3: CrossVB= %f %f %f   mag-1= %g \n", dirCrossVB[0], dirCrossVB[1], dirCrossVB[2],
         dirCrossVB.Mag()-1.0 );
  // dirCrossVB = dirCrossVB.Unit();
  printf("CVFHS> Dot products   d1.d2= %g   d2.d3= %g  d3.d1= %g \n",
         dir1Field.Dot(dirVelX), dirVelX.Dot( dirCrossVB), dirCrossVB.Dot(dir1Field) );
   ***/
  REQUIRE( abs(dot_product(dir1Field, dirVelX)) < 1.e-6);
  REQUIRE( abs(dot_product(dirVelX, dirCrossVB)) < 1.e-6);
  REQUIRE( abs(dot_product(dirCrossVB, dir1Field)) < 1.e-6);

  Real_v phi = -step * charge * fBmag * kB2C_local / momentum;
  std::cout<<" ConstFieldHelixStepper DoStep(): spot 3: phi="<< phi <<", "<< momentum <<", step="<< step <<"\n";

  // printf("CVFHS> phi= %g \n", vecCore::Get(phi,0) );  // phi (scalar)  or phi[0] (vector)

  Real_v cosphi;                 //  = Cos(phi);
  Real_v sinphi;                 //  = Sin(phi);
  sincos(phi, &sinphi, &cosphi); // Opportunity for new 'efficient' method !?

  //endPosition = startPosition + R * (cosphi - 1) * dirCrossVB - R * sinphi * dirVelX +
  //              step * UVdotUB * dir1Field; //   'Drift' along field direction
  endPosition = startPosition;
  axpy(R * (cosphi - 1.0), dirCrossVB, &endPosition);
  axpy(-R * sinphi, dirVelX, &endPosition);
  axpy(step * UVdotUB, dir1Field, &endPosition);
  std::cout<<" ConstFieldHelixStepper DoStep(): spot 4: endPos="<< endPosition <<", "<< sinphi <<"\n";

  // dx = dx0 * cosphi - sinphi * dy0;
  // dy = dy0 * cosphi + sinphi * dx0;
  // dz = dz0;
  // printf(" phi= %f, sin(phi)= %f , sin(V,B)= %f\n", phi, sinphi, sinVB );
  //endDirection = UVdotUB * dir1Field + cosphi * sinVB * dirVelX + sinphi * sinVB * dirCrossVB;
  endDirection = {0, 0, 0};
  axpy(UVdotUB, dir1Field, &endDirection);
  axpy(cosphi * sinVB, dirVelX, &endDirection);
  axpy(sinphi * sinVB, dirCrossVB, &endDirection);
}

// /**
//  * basket version of dostep
//  */

// //  SW: commented out due to explicit Vc dependence and since it is not currently used
// //       leaving the code here to show how one would dispatch to the kernel with Vc
// #define _R_ __restrict__

// template <typename Real_v>
// void ConstFieldHelixStepper::DoStepArr(double const *_R_ posx, double const *_R_ posy, double const *_R_ posz,
//                                        double const *_R_ dirx, double const *_R_ diry, double const *_R_ dirz,
//                                        double const *_R_ charge, double const *_R_ momentum, double const *_R_ step,
//                                        double *_R_ newposx, double *_R_ newposy, double *_R_ newposz,
//                                        double *_R_ newdirx, double *_R_ newdiry, double *_R_ newdirz, int np) const
// {
//   const size_t vectorSize = vecCore::VectorSize<Real_v>();
//   using vecCore::Load;
//   using vecCore::Set;
//   using vecCore::Store;

//   // std::cout << " --- ConstFieldHelixStepper::DoStepArr called." << std::endl;

//   int i;
//   for (i = 0; i < np; i += vectorSize) {
//     // results cannot not be temporaries
//     //    Real3 newPosition, newDirection;
//     Real_v oldPosx_v, oldPosy_v, oldPosz_v, oldDirx_v, oldDiry_v, oldDirz_v;
//     Real_v newposx_v, newposy_v, newposz_v, newdirx_v, newdiry_v, newdirz_v;
//     Real_v charge_v, momentum_v, stepSz_v;

//     Load(oldPosx_v, &posx[i]);
//     Load(oldPosy_v, &posy[i]);
//     Load(oldPosz_v, &posz[i]);
//     Load(oldDirx_v, &dirx[i]);
//     Load(oldDiry_v, &diry[i]);
//     Load(oldDirz_v, &dirz[i]);
//     Load(charge_v, &charge[i]);
//     Load(momentum_v, &momentum[i]);
//     Load(stepSz_v, &step[i]);

//     // This check should be optional
//     REQUIRE(abs(oldDirx_v*oldDirx_v + oldDiry_v*oldDiry_v + oldDirz_v*oldDirz_v - 1.0) < 1.0e-6);

//     DoStep<Real_v>(oldPosx_v, oldPosy_v, oldPosz_v, oldDirx_v, oldDiry_v, oldDirz_v, charge_v, momentum_v, stepSz_v,
//                    newposx_v, newposy_v, newposz_v, newdirx_v, newdiry_v, newdirz_v);

//     REQUIRE(abs(newDirx_v*newDirx_v + newDiry_v*newDiry_v + newDirz_v*newDirz_v - 1.0) < 1.0e-6);

//     // write results
//     Store(newposx_v, &newposx[i]);
//     Store(newposy_v, &newposy[i]);
//     Store(newposz_v, &newposz[i]);
//     Store(newdirx_v, &newdirx[i]);
//     Store(newdiry_v, &newdiry[i]);
//     Store(newdirz_v, &newdirz[i]);
//   }

//   // tail part
//   for (; i < np; i++)
//     DoStep<double>(posx[i], posy[i], posz[i], dirx[i], diry[i], dirz[i], charge[i], momentum[i], step[i], newposx[i],
//                    newposy[i], newposz[i], newdirx[i], newdiry[i], newdirz[i]);
// }

//________________________________________________________________________________
template <typename Real_v>
CELER_FORCEINLINE_FUNCTION void ConstFieldHelixStepper::PrintStep(Real3 const &startPosition,
                                                          Real3 const &startDirection,
                                                          Real_v const &charge, Real_v const &momentum,
                                                          Real_v const &step, Real3 &endPosition,
                                                          Real3 &endDirection) const
{
  // Debug printing of input & output
  printf(" HelixSteper::PrintStep \n");
  Real_v x0, y0, z0, dirX0, dirY0, dirZ0;
  Real_v x, y, z, dx, dy, dz;
  x0    = startPosition[0];
  y0    = startPosition[1];
  z0    = startPosition[2];
  dirX0 = startDirection[0];
  dirY0 = startDirection[1];
  dirZ0 = startDirection[2];
  x     = endPosition[0];
  y     = endPosition[1];
  z     = endPosition[2];
  dx    = endDirection[0];
  dy    = endDirection[1];
  dz    = endDirection[2];

  printf("Start> Pos= %8.5f %8.5f %8.5f  Dir= %8.5f %8.5f %8.5f ", x0, y0, z0, dirX0, dirY0, dirZ0);
  printf(" step= %10.6f ", step / 10.0);     // / units::mm );
  printf(" q= %3.1f ", charge / 10.0);    // in e+ units ?
  printf(" p= %10.6f ", momentum / 10.0); // / units::GeV );
  // printf(" ang= %7.5f ", angle );
  printf(" End> Pos= %9.6f %9.6f %9.6f  Mom= %9.6f %9.6f %9.6f\n", x, y, z, dx, dy, dz);
}

} // namespace celeritas
