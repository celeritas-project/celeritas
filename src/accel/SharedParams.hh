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

class G4ParticleDefinition;

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace detail
{
class HitManager;
class OffloadWriter;
}  // namespace detail

class CoreParams;
class CoreStateInterface;
struct Primary;
struct SetupOptions;
class StepCollector;
class GeantGeoParams;
class OutputRegistry;

//---------------------------------------------------------------------------//
/*!
 * Shared (one instance for all threads) Celeritas problem data.
 *
 * The \c CeleritasDisabled accessor queries the \c CELER_DISABLE environment
 * variable as a global option for disabling Celeritas offloading. This is
 * implemented by \c SimpleOffload
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
    using SPConstParams = std::shared_ptr<CoreParams const>;
    using VecG4ParticleDef = std::vector<G4ParticleDefinition*>;
    //!@}

    //! Setup for Celeritas usage
    enum class Mode
    {
        uninitialized,
        disabled,
        kill_offload,
        enabled,
        size_
    };

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

    //!@}
    //!@{
    //! \name Construction

    // Construct in an uninitialized state
    SharedParams() = default;

    // Construct Celeritas using Geant4 data on the master thread.
    explicit SharedParams(SetupOptions const& options);

    // Construct for output only
    explicit SharedParams(std::string output_filename);

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
    inline SPConstParams Params() const;

    // Get a vector of particles supported by Celeritas offloading
    inline VecG4ParticleDef const& OffloadParticles() const;

    //! Whether the class has been constructed
    explicit operator bool() const { return mode_ != Mode::uninitialized; }

    //!@}
    //!@{
    //! \name Internal use only

    using SPHitManager = std::shared_ptr<detail::HitManager>;
    using SPOffloadWriter = std::shared_ptr<detail::OffloadWriter>;
    using SPOutputRegistry = std::shared_ptr<OutputRegistry>;
    using SPState = std::shared_ptr<CoreStateInterface>;
    using SPConstGeantGeoParams = std::shared_ptr<GeantGeoParams const>;

    //! Initialization status and integration mode
    Mode mode() const { return mode_; }

    // Hit manager, to be used only by LocalTransporter
    inline SPHitManager const& hit_manager() const;

    // Optional offload writer, only for use by LocalTransporter
    inline SPOffloadWriter const& offload_writer() const;

    // Output registry
    inline SPOutputRegistry const& output_reg() const;

    // Let LocalTransporter register the thread's state
    void set_state(unsigned int stream_id, SPState&&);

    // Number of streams, lazily obtained from run manager
    unsigned int num_streams() const;

    // Geant geometry wrapper, lazily created
    SPConstGeantGeoParams const& geant_geo_params() const;
    //!@}

  private:
    //// DATA ////

    // Created during initialization
    Mode mode_{Mode::uninitialized};
    std::shared_ptr<CoreParams> params_;
    std::shared_ptr<detail::HitManager> hit_manager_;
    std::shared_ptr<StepCollector> step_collector_;
    VecG4ParticleDef particles_;
    std::string output_filename_;
    SPOffloadWriter offload_writer_;
    std::vector<std::shared_ptr<CoreStateInterface>> states_;

    // Lazily created
    SPOutputRegistry output_reg_;
    SPConstGeantGeoParams geant_geo_;

    //// HELPER FUNCTIONS ////

    void initialize_core(SetupOptions const& options);
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
auto SharedParams::Params() const -> SPConstParams
{
    CELER_EXPECT(mode_ == Mode::enabled);
    CELER_ENSURE(params_);
    return params_;
}

//---------------------------------------------------------------------------//
/*!
 * Get a vector of particles supported by Celeritas offloading.
 */
auto SharedParams::OffloadParticles() const -> VecG4ParticleDef const&
{
    CELER_EXPECT(*this);
    return particles_;
}

//---------------------------------------------------------------------------//
/*!
 * Hit manager, to be used only by LocalTransporter.
 *
 * If sensitive detector callback is disabled, the hit manager will be null.
 */
auto SharedParams::hit_manager() const -> SPHitManager const&
{
    CELER_EXPECT(*this);
    return hit_manager_;
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
}  // namespace celeritas
