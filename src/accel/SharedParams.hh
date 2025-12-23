//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SharedParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "corecel/Assert.hh"
#include "geocel/BoundingBox.hh"

#include "Types.hh"

class G4ParticleDefinition;
class G4VPhysicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace detail
{
class OffloadWriter;
}  // namespace detail

namespace optical
{
class Transporter;
}  // namespace optical

class ActionSequence;
class CoreParams;
class CoreStateInterface;
class GeantGeoParams;
class GeantSd;
class OpticalCollector;
class OutputRegistry;
class StepCollector;
class TimeOutput;
struct Primary;
struct SetupOptions;

//---------------------------------------------------------------------------//
/*!
 * Shared (one instance for all threads) Celeritas problem data.
 *
 * The \c CeleritasDisabled accessor queries the \c CELER_DISABLE environment
 * variable as a global option for disabling Celeritas offloading.
 *
 * This should be instantiated on the master thread during problem setup,
 * preferably as a shared pointer. The shared pointer should be
 * passed to a thread-local \c LocalTransporter instance. At the beginning of
 * the run, after Geant4 has initialized physics data, the \c Initialize method
 * must be called first on the "master" thread to populate the Celeritas data
 * structures (geometry, physics). \c InitializeWorker must subsequently be
 * invoked on all worker threads to set up thread-local data (specifically,
 * CUDA device initialization).
 *
 * Some low-level objects, such as the output diagnostics and Geant4 geometry
 * wrapper, can be created independently of Celeritas being enabled.
 */
class SharedParams
{
  public:
    //!@{
    //! \name Type aliases
    using SPParams = std::shared_ptr<CoreParams>;
    using SPConstParams = std::shared_ptr<CoreParams const>;
    using VecG4PD = std::vector<G4ParticleDefinition*>;
    using Mode = OffloadMode;
    //!@}

  public:
    //!@{
    //! \name Status

    // Whether celeritas is disabled, set to kill, or to be enabled
    static Mode GetMode();

    // True if Celeritas is globally disabled using the CELER_DISABLE env
    // Remove in 0.7
    [[deprecated]]
    static bool CeleritasDisabled();

    // Whether to kill tracks that would have been offloaded
    // Remove in 0.7
    [[deprecated]]
    static bool KillOffloadTracks();

    // Get list of all supported particles in Celeritas
    static VecG4PD const& supported_offload_particles();

    // Get list of enabled particles for offloading by default
    static VecG4PD const& default_offload_particles();
    //!@}
    //!@{
    //! \name Construction

    // Construct in an uninitialized state
    SharedParams() = default;

    // Construct Celeritas using Geant4 data on the master thread.
    explicit SharedParams(SetupOptions const& options);

    // Initialize shared data on the "master" thread
    inline void Initialize(SetupOptions const& options);

    // On worker threads, set up data with thread storage duration
    void InitializeWorker(SetupOptions const& options);

    // Write (shared) diagnostic output and clear shared data on master
    void Finalize();

    //!@}
    //!@{
    //! \name Accessors

    // Access constructed Celeritas data
    inline SPParams const& Params();

    // Access constructed Celeritas data
    inline SPConstParams Params() const;

    // Get a vector of particles to be used by Celeritas offloading
    inline VecG4PD const& OffloadParticles() const;

    //! Whether the class has been constructed
    explicit operator bool() const { return mode_ != Mode::uninitialized; }

    //!@}
    //!@{
    //! \name Internal use only

    using SPActionSequence = std::shared_ptr<ActionSequence>;
    using SPGeantSd = std::shared_ptr<GeantSd>;
    using SPOffloadWriter = std::shared_ptr<detail::OffloadWriter>;
    using SPOutputRegistry = std::shared_ptr<OutputRegistry>;
    using SPTimeOutput = std::shared_ptr<TimeOutput>;
    using SPState = std::shared_ptr<CoreStateInterface>;
    using SPOpticalCollector = std::shared_ptr<OpticalCollector>;
    using SPOpticalTransporter = std::shared_ptr<optical::Transporter>;
    using SPConstGeantGeoParams = std::shared_ptr<GeantGeoParams const>;
    using BBox = BoundingBox<double>;

    //! Initialization status and integration mode
    Mode mode() const { return mode_; }

    // Optical properties (only if using optical physics)
    inline SPOpticalCollector const& optical_collector() const;

    // Optical params (only if using optical physics)
    inline SPOpticalTransporter const& optical_transporter() const;

    // Action sequence
    inline SPActionSequence const& actions() const;

    // Hit manager, to be used only by LocalTransporter
    inline SPGeantSd const& hit_manager() const;

    // Optional offload writer, only for use by LocalTransporter
    inline SPOffloadWriter const& offload_writer() const;

    // Output registry
    inline SPOutputRegistry const& output_reg() const;

    // Access the timer
    inline SPTimeOutput const& timer() const;

    // Let LocalTransporter register the thread's state
    void set_state(unsigned int stream_id, SPState&&);

    // Number of streams, lazily obtained from run manager
    unsigned int num_streams() const;

    // Geometry bounding box (CLHEP units)
    BBox const& bbox() const { return bbox_; }
    //!@}

  private:
    //// DATA ////

    // Created during initialization
    Mode mode_{Mode::uninitialized};
    SPConstGeantGeoParams geant_geo_;
    std::shared_ptr<CoreParams> params_;
    SPOpticalCollector optical_collector_;
    SPOpticalTransporter optical_transporter_;
    SPActionSequence actions_;
    std::shared_ptr<GeantSd> geant_sd_;
    std::shared_ptr<StepCollector> step_collector_;
    VecG4PD offload_particles_;
    std::string output_filename_;
    SPOffloadWriter offload_writer_;
    std::vector<std::shared_ptr<CoreStateInterface>> states_;
    SPOutputRegistry output_reg_;
    SPTimeOutput timer_;
    BBox bbox_;

    //// HELPER FUNCTIONS ////

    void set_num_streams(unsigned int num_streams);
    void try_output() const;
};

//---------------------------------------------------------------------------//
/*!
 * Helper for making initialization more obvious from user code.
 */
void SharedParams::Initialize(SetupOptions const& options)
{
    *this = SharedParams(options);
}

//---------------------------------------------------------------------------//
/*!
 * Access Celeritas data.
 *
 * This can only be called after \c Initialize.
 */
auto SharedParams::Params() -> SPParams const&
{
    CELER_EXPECT(mode_ == Mode::enabled);
    CELER_ENSURE(params_);
    return params_;
}

//---------------------------------------------------------------------------//
/*!
 * Access Celeritas data.
 *
 * This can only be called after \c Initialize.
 */
auto SharedParams::Params() const -> SPConstParams
{
    CELER_EXPECT(mode_ == Mode::enabled);
    CELER_ENSURE(params_);
    return params_;
}

//---------------------------------------------------------------------------//
/*!
 * Get a vector of particles to be used by Celeritas offloading.
 */
auto SharedParams::OffloadParticles() const -> VecG4PD const&
{
    CELER_EXPECT(*this);
    return offload_particles_;
}

//---------------------------------------------------------------------------//
/*!
 * Optical transporter: null if Celeritas optical physics is disabled.
 */
auto SharedParams::optical_transporter() const -> SPOpticalTransporter const&
{
    CELER_EXPECT(*this);
    return optical_transporter_;
}

//---------------------------------------------------------------------------//
/*!
 * Optical data: null if Celeritas optical physics is disabled.
 */
auto SharedParams::optical_collector() const -> SPOpticalCollector const&
{
    CELER_EXPECT(*this);
    return optical_collector_;
}

//---------------------------------------------------------------------------//
/*!
 * Hit manager, to be used only by LocalTransporter.
 *
 * If sensitive detector callback is disabled, the hit manager will be null.
 */
auto SharedParams::hit_manager() const -> SPGeantSd const&
{
    CELER_EXPECT(*this);
    return geant_sd_;
}

//---------------------------------------------------------------------------//
/*!
 * Action sequence for the stepper.
 */
auto SharedParams::actions() const -> SPActionSequence const&
{
    CELER_EXPECT(*this);
    return actions_;
}

//---------------------------------------------------------------------------//
/*!
 * Optional offload writer, only for use by LocalTransporter.
 */
auto SharedParams::offload_writer() const -> SPOffloadWriter const&
{
    CELER_EXPECT(*this);
    return offload_writer_;
}

//---------------------------------------------------------------------------//
/*!
 * Output registry for writing data at end of run.
 */
auto SharedParams::output_reg() const -> SPOutputRegistry const&
{
    CELER_EXPECT(*this);
    return output_reg_;
}

//---------------------------------------------------------------------------//
/*!
 * Access the timer.
 */
auto SharedParams::timer() const -> SPTimeOutput const&
{
    CELER_EXPECT(*this);
    CELER_EXPECT(timer_);
    return timer_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
