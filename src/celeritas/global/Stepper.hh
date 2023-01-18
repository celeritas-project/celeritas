//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoParamsFwd.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/random/RngParamsFwd.hh"
#include "celeritas/track/TrackInitData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;
struct Primary;

namespace detail
{
class ActionSequence;
}

//---------------------------------------------------------------------------//
/*!
 * State-specific options for the stepper.
 *
 * - \c params : Problem definition
 * - \c num_track_slots : Maximum number of threads to run in parallel on GPU
 * - \c sync : Whether to synchronize device between actions
 */
struct StepperInput
{
    std::shared_ptr<CoreParams const> params;
    size_type num_track_slots{};
    bool sync{false};

    //! True if defined
    explicit operator bool() const { return params && num_track_slots > 0; }
};

//---------------------------------------------------------------------------//
/*!
 * Track counters for a step.
 */
struct StepperResult
{
    size_type queued{};  //!< Pending track initializers at end of step
    size_type active{};  //!< Active tracks at start of step
    size_type alive{};  //!< Active and alive at end of step

    //! True if more steps need to be run
    explicit operator bool() const { return queued > 0 || alive > 0; }
};

//---------------------------------------------------------------------------//
//! Interface class for stepper classes.
class StepperInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Input = StepperInput;
    using ActionSequence = detail::ActionSequence;
    using SpanConstPrimary = Span<Primary const>;
    using result_type = StepperResult;
    //!@}

  public:
    // Transport existing states
    virtual StepperResult operator()() = 0;

    // Transport existing states and these new primaries
    virtual StepperResult operator()(SpanConstPrimary primaries) = 0;

    //! Whether the stepper is assigned/valid
    virtual explicit operator bool() const = 0;

    //! Get action sequence for timing diagnostics
    virtual ActionSequence const& actions() const = 0;

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~StepperInterface() = default;
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
class Stepper final : public StepperInterface
{
  public:
    // Construct with problem parameters and setup options
    explicit Stepper(Input input);

    //!@{
    //! Prohibit copying but allow moving and empty construction
    Stepper() = default;
    Stepper(Stepper&&) = default;
    Stepper(Stepper const&) = delete;
    Stepper& operator=(Stepper&&) = default;
    Stepper& operator=(Stepper const&) = delete;
    //!@}

    // Default destructor
    ~Stepper();

    // Transport existing states
    StepperResult operator()() final;

    // Transport existing states and these new primaries
    StepperResult operator()(SpanConstPrimary primaries) final;

    //! Whether the stepper is assigned/valid
    explicit operator bool() const final { return static_cast<bool>(states_); }

    //! Get action sequence for timing diagnostics
    ActionSequence const& actions() const final { return *actions_; }

  private:
    // Params and call sequence
    std::shared_ptr<CoreParams const> params_;
    std::shared_ptr<detail::ActionSequence> actions_;

    // State data
    CollectionStateStore<CoreStateData, M> states_;

    // Combined param/state for action calls
    CoreRef<M> core_ref_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
