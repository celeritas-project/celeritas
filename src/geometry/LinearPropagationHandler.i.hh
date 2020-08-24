//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagation.i.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geometry/LinearPropagationHandler.hh"
//#include "Geant/geometry/NavigationInterface.hpp"
#include "base/ArrayUtils.hh"
#include "geometry/Types.hh"

namespace celeritas {

//______________________________________________________________________________
// Do straight propagation to physics process or boundary
// Return false if problems (track stuck at boundary after 10 steps), or true otherwise.
// 
CELER_FORCEINLINE_FUNCTION
void LinearPropagationHandler::quickLinearStep(GeoTrackView &track, real_type step)
{
  axpy(step, track.dir(), &track.pos());
  track.step() += step;
}

CELER_FORCEINLINE_FUNCTION
void LinearPropagationHandler::commitStepUpdates(GeoTrackView &track)
{
  // other track step updates -- are all those needed?
  track.pstep() -= track.step();
  track.total_length() += track.step();
  track.proper_time() += track.step() / (track.beta() * constants::cLight);
  track.snext() -= track.step();

  if (track.snext ()> 1.e-8 || track.safety() < 1.e-8 ) {
    std::cout<<"***** LinearPropHandler::Propagate: forcing safety or snext to zero: snext="<< track.snext()
	     <<", safety="<< track.safety() << std::endl;
  }
  if (track.snext() < 1.E-8) track.snext() = 0.0;
  if (track.safety() < 1.E-8) track.safety() = 0.0;

  track.has_same_path();  // ensure this is called at least once to update _vgstate
  if ( track.next_exiting() ) {
    track.status() = GeoTrackStatus::ExitingSetup;
  }
}

CELER_FORCEINLINE_FUNCTION
bool LinearPropagationHandler::Propagate(GeoTrackView &track)
{
  // Scalar geometry length computation. The track is moved along track.dir() direction
  // by a distance track.snext()
  real_type step = track.snext();
  std::cout<<" LinearPropHandle::Propagate(): step() = "<< &(track.step()) << std::endl;
  track.step() = 0;
  quickLinearStep(track, step);

  // Update total number of steps
  int nsmall = 0;
  bool status = true;
  if (track.boundary() == Boundary::Yes) {
    track.status() = GeoTrackStatus::Boundary;
    // Find out location after boundary
    while ( not_at_boundary(track) && track.has_same_path() ) {
      quickLinearStep(track, 1.E-4 * units::mm);
      nsmall++;
      if (nsmall > 10) {
        // Most likely a nasty overlap, some smarter action required. For now, just
        // kill the track.

	// Log(kError) << "LinearPropagator: track " << track.fHistoryState.fParticle
        //             << " from event " << track.fHistoryState.fEvent
        //             << " stuck -> killing it";
 
        // Deposit track energy, then go directly to stepping actions
        // track->Stop();
	track.setKineticEnergy(0.0);
        track.status() = GeoTrackStatus::Killed;
        // jump to track->SetStage(kSteppingActionsStage);
        // record number of killed particles: td->fNkilled++;
        status = false;
        break;
      }
    }
  }
  else {
    track.status() = GeoTrackStatus::Physics;
    // Update number of steps to physics
    //td->fNphys++;
  }

  commitStepUpdates(track);
  return status;
}

//______________________________________________________________________________
CELER_FORCEINLINE_FUNCTION
bool LinearPropagationHandler::not_at_boundary(GeoTrackView const& track) const
{
  static constexpr real_type boundary_tolerance = 1.0e-6 * units::mm;
  return ( track.safety() > boundary_tolerance && track.snext() > boundary_tolerance ); 
}

} // namespace celeritas
