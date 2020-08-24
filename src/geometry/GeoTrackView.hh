//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTrackView.hh
//---------------------------------------------------------------------------//
#ifndef geometry_GeoTrackView_hh
#define geometry_GeoTrackView_hh

#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/navigation/NavigationState.h>

#include "base/Constants.hh"
#include "base/Macros.hh"
#include "base/NumericLimits.hh"
#include "GeoStatePointers.hh"
#include "GeoParamsPointers.hh"
#include "Types.hh"
#include "detail/VGCompatibility.hh"

#include "base/ArrayIO.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Operate on the device with shared (persistent) data and local state.
 *
 * \code
    GeoTrackView geom(vg_view, vg_state_view, thread_id);
   \endcode
 */
class GeoTrackView
{
  public:
    struct Initializer_t
    {
        Real3 pos;
        Real3 dir;
        real_type mass;
        real_type energy;
        real_type momentum;
        real_type time;
        real_type totlen;
        real_type safety;
        size_type num_steps;
        GeoTrackStatus status;
    };

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION GeoTrackView(const GeoParamsPointers& data,
                                       const GeoStatePointers&  stateview,
                                       const ThreadId&          id);

    // Initialize the state
    inline CELER_FUNCTION GeoTrackView& operator=(const Initializer_t& init);

    CELER_FORCEINLINE_FUNCTION void setPosition(const Real3& newpos) {
      pos_[0] = newpos[0];
      pos_[1] = newpos[1];
      pos_[2] = newpos[2];
    }
    CELER_FORCEINLINE_FUNCTION void setDirection(const Real3& newdir) {
      dir_[0] = newdir[0];
      dir_[1] = newdir[1];
      dir_[2] = newdir[2];
    }

    CELER_FUNCTION void setEnergy(real_type energy);
    CELER_FORCEINLINE_FUNCTION void setKineticEnergy(real_type kinEnergy);
    CELER_FORCEINLINE_FUNCTION void setMomentum(real_type momentum);

    // Find the distance to the next boundary
    inline CELER_FUNCTION void find_next_step();
    // Move to the next boundary
    inline CELER_FUNCTION void move_next_step();
    // Check for boundary cross in step, update next state if needed
    CELER_FUNCTION bool has_same_path();

    //@{
    //! State accessors
    CELER_FUNCTION const Real3&    pos() const { return pos_; }
    CELER_FUNCTION const Real3&    dir() const { return dir_; }
    CELER_FUNCTION       Real3&    pos()       { return pos_; }
    CELER_FUNCTION       Real3&    dir()       { return dir_; }
    CELER_FUNCTION real_type       next_step() const { return next_step_; }
    CELER_FORCEINLINE_FUNCTION VolumeId volume_id() const;
    CELER_FORCEINLINE_FUNCTION Boundary boundary() const;
    CELER_FORCEINLINE_FUNCTION Boundary next_boundary() const;
    CELER_FORCEINLINE_FUNCTION bool     next_exiting();

    CELER_FORCEINLINE_FUNCTION GeoTrackStatus const& status() const { return status_; }
    CELER_FORCEINLINE_FUNCTION GeoTrackStatus& status() { return status_; }
    CELER_FORCEINLINE_FUNCTION size_type& num_steps() const { return num_steps_; }
    CELER_FORCEINLINE_FUNCTION real_type& proper_time() const { return proper_time_; }
    CELER_FORCEINLINE_FUNCTION real_type& total_length() const { return total_length_; }

    CELER_FORCEINLINE_FUNCTION real_type const& safety() const { return safety_; }
    CELER_FORCEINLINE_FUNCTION real_type const& step()   const { return step_;   }
    CELER_FORCEINLINE_FUNCTION real_type const& pstep()  const { return pstep_;  }
    CELER_FORCEINLINE_FUNCTION real_type const& snext()  const { return snext_;  }
    CELER_FORCEINLINE_FUNCTION real_type& safety() { return safety_; }
    CELER_FORCEINLINE_FUNCTION real_type& step()   { return step_;   }
    CELER_FORCEINLINE_FUNCTION real_type& pstep()  { return pstep_;  }
    CELER_FORCEINLINE_FUNCTION real_type& snext()  { return snext_;  }

    // particle params -- temporarily here until available from elsewhere
    CELER_FORCEINLINE_FUNCTION const real_type mass() const {
      //return mass_;
      return restEnergy() / (constants::cLight * constants::cLight);
    }
    CELER_FORCEINLINE_FUNCTION size_type charge() const { return 1; }

    // dynamic variables -- temporarily here until available from elsewhere
    CELER_FORCEINLINE_FUNCTION const real_type& energy() const { return energy_; }
    CELER_FORCEINLINE_FUNCTION const real_type& momentum() const { return momentum_; }

    CELER_FORCEINLINE_FUNCTION real_type beta() const { return momentum_ / energy_; }
    CELER_FORCEINLINE_FUNCTION real_type gamma() const { return energy_ / restEnergy(); }

    CELER_FORCEINLINE_FUNCTION real_type kineticEnergy() const { return restEnergy() * (gamma() - 1.0); }
    CELER_FUNCTION real_type restEnergy() const
    {
      //real_type restE = mass_ * celeritas::constants::cLight * celeritas::constants::cLight;
      real_type restE = constants::electron_mass_c2;
      std::cout <<"restEnergy(): &thisGTV="<< this <<": mass/kg="<< mass_/units::kg <<" @"<< &mass_
		<<", mass/(GeV)="<< mass_/units::GeV
		<<", mass/(GeV/cˆ2)="<< mass_/(units::GeV/constants::cLight/constants::cLight)
		<<", restE/GeV = " << restE / units::GeV
		<< std::endl;
      return restE;
    }

    friend std::ostream& operator<<(std::ostream& os, const GeoTrackView& t)
    {
      os <<" addr="<< (void*)&t
	 <<", pos="<< t.pos()
	 <<", dir="<< t.dir()
	 <<", status="<< t.status()
	 <<", #steps="<< t.num_steps()
	// <<", step="<< t.
	// <<", Pstep="<< t.
	// <<", snext="<< t.
	 <<", safety="<< t.safety()
	 <<", time="<< t.proper_time()
	 <<", totLen="<< t.total_length()
	 <<"\n";
      t.vgstate_.Print();
      t.vgnext_.Print();
      return os;
    }

    //@}

  private:
    //@{
    //! Type aliases
    using Volume   = vecgeom::VPlacedVolume;
    using NavState = vecgeom::NavigationState;
    //@}

    //! Shared/persistent geometry data
    const GeoParamsPointers& shared_;

    //@{
    //! Referenced thread-local data
    NavState&  vgstate_;
    NavState&  vgnext_;
    Real3&     pos_;
    Real3&     dir_;
    real_type& next_step_;

    real_type& mass_;
    real_type& energy_;
    real_type& momentum_;

    real_type& proper_time_;
    real_type& total_length_;

    real_type& safety_;
    real_type& step_;
    real_type& pstep_;
    real_type& snext_;
    size_type& num_steps_;
    GeoTrackStatus& status_;
    //@}

  private:
    // Get a reference to the state from a NavStatePool's pointer
    static inline CELER_FUNCTION NavState&
                                 get_nav_state(void* state, int vgmaxdepth, ThreadId thread);

    // Get a reference to the current volume
    inline CELER_FUNCTION const Volume& volume() const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "GeoTrackView.i.hh"

#endif // geometry_GeoTrackView_hh
