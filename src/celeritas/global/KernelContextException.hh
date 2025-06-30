//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/KernelContextException.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "corecel/Assert.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "CoreTrackDataFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreTrackView;
struct JsonPimpl;

//---------------------------------------------------------------------------//
/*!
 * Provide contextual information about failed errors on CPU.
 *
 * When a CPU track hits an exception, gather properties about the current
 * thread and failing track. These properties are accessible through this
 * exception class *or* they can be chained into the failing exception and
 * processed by \c ExceptionOutput as context for the failure.
 *
 * \code
    CELER_TRY_HANDLE_CONTEXT(
        step(ThreadId{i}),
        capture_exception,
        KernelContextException(data.params, data.states, ThreadId{i},
 this->label())
    );
 * \endcode
 */
class KernelContextException : public RichContextException
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with track data and kernel label
    KernelContextException(HostCRef<CoreParamsData> const& params,
                           HostRef<CoreStateData> const& states,
                           ThreadId tid,
                           std::string_view label);

    // This class type
    char const* type() const final;

    // Save context to a JSON object
    void output(JsonPimpl* json) const final;

    //! Get an explanatory message
    char const* what() const noexcept final { return what_.c_str(); }

    //!@{
    //! \name Track accessors

    //! Kernel thread ID
    ThreadId thread() const { return thread_; }
    //! Track slot ID
    TrackSlotId track_slot() const { return track_slot_; }
    //! Event ID
    EventId event() const { return event_; }
    //! Track ID
    TrackId track() const { return track_; }
    //! Parent track ID
    TrackId parent() const { return parent_; }
    //! Step counter
    size_type num_steps() const { return num_steps_; }
    //! Particle type
    ParticleId particle() const { return particle_; }
    //! Particle energy
    Energy energy() const { return energy_; }
    //! Position
    Real3 const& pos() const { return pos_; }
    //! Direction
    Real3 const& dir() const { return dir_; }
    //! Volume ID
    VolumeId volume() const { return volume_; }
    //! Surface
    ImplSurfaceId surface() const { return surface_; }
    //!@}

    //! Label of the kernel that died
    std::string const& label() const { return label_; }

  private:
    ThreadId thread_;
    TrackSlotId track_slot_;
    EventId event_;
    TrackId track_;
    TrackId parent_;
    size_type num_steps_;
    ParticleId particle_;
    Energy energy_;
    Real3 pos_;
    Real3 dir_;
    VolumeId volume_;
    ImplSurfaceId surface_;

    std::string label_;
    std::string what_;

    // Populate properties during construction
    void initialize(CoreTrackView const& core);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
