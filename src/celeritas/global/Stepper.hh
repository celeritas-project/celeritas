//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoParamsFwd.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/random/RngParamsFwd.hh"
#include "celeritas/track/TrackInitData.hh"

namespace celeritas
{
class CoreParams;
struct Primary;
//---------------------------------------------------------------------------//
/*!
 * State-specific options for the stepper.
 *
 * - \c params : Problem definition
 * - \c num_track_slots : Maximum number of threads to run in parallel on GPU
 * - \c num_initializers : Maximum number of secondaries + primaries allowable
 */
struct StepperInput
{
    std::shared_ptr<const CoreParams> params;
    size_type                         num_track_slots{};
    size_type                         num_initializers{};

    // TODO: integrate into action interface
    ActionId post_step_callback;
};

//---------------------------------------------------------------------------//
/*!
 * Track counters for a step.
 */
struct StepperResult
{
    size_type queued{}; //!< Pending track initializers at end of step
    size_type active{}; //!< Active tracks at start of step
    size_type alive{};  //!< Active and alive at end of step

    //! True if more steps need to be run
    explicit operator bool() const
    {
        return queued > 0 || active > 0 || alive > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Manage a state vector and execute a single step on all of them.
 *
 * \code
   Stepper<MemSpace::host> step(input);

   // Transport primaries for the initial step
   StepperResult alive_tracks = step(my_primaries);
   while (alive_tracks)
   {
       // Transport secondaries
       alive_tracks = step();
   }
   \endcode
 */
template<MemSpace M>
class Stepper
{
  public:
    //!@{
    //! Type aliases
    using Input       = StepperInput;
    using VecPrimary  = std::vector<Primary>;
    using result_type = StepperResult;
    //!@}

  public:
    // Construct with problem parameters and setup options
    explicit Stepper(Input input);

    //!@{
    //! Prohibit copying but allow moving and empty construction
    Stepper()               = default;
    Stepper(Stepper&&)      = default;
    Stepper(const Stepper&) = delete;
    Stepper& operator=(Stepper&&) = default;
    Stepper& operator=(const Stepper&) = delete;
    //!@}

    // Default destructor
    ~Stepper();

    // Transport existing states
    StepperResult operator()();

    // Transport existing states and these new primaries
    StepperResult operator()(VecPrimary primaries);

    //! Whether the stepper is assigned/valid
    explicit operator bool() const { return static_cast<bool>(states_); }

  private:
    // Params and call sequence
    std::shared_ptr<const CoreParams> params_;
    std::vector<ActionId>             actions_;

    // State data
    size_type                               num_initializers_;
    CollectionStateStore<CoreStateData, M>  states_;
    TrackInitStateData<Ownership::value, M> inits_;

    // Combined param/state for action calls
    CoreRef<M> core_ref_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
