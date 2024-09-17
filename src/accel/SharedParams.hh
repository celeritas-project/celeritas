//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
    using VecG4ParticleDef = std::vector<G4ParticleDefinition const*>;
    //!@}

  public:
    //!@{
    //! \name Construction

    // True if Celeritas is globally disabled using the CELER_DISABLE env
    static bool CeleritasDisabled();

    // Whether to kill tracks that would have been offloaded
    static bool KillOffloadTracks();

    // Construct in an uninitialized state
    SharedParams() = default;

    // Construct Celeritas using Geant4 data on the master thread.
    explicit SharedParams(SetupOptions const& options);

    // Construct for output only
    explicit SharedParams(std::string output_filename);

    // Initialize shared data on the "master" thread
    inline void Initialize(SetupOptions const& options);

    // Initialize shared data on the "master" thread
    inline void Initialize(std::string output_filename);

    // On worker threads, set up data with thread storage duration
    static void InitializeWorker(SetupOptions const& options);

    // Write (shared) diagnostic output and clear shared data on master
    void Finalize();

    //!@}
    //!@{
    //! \name Accessors

    // Access constructed Celeritas data
    inline SPConstParams Params() const;

    // Get a vector of particles supported by Celeritas offloading
    VecG4ParticleDef const& OffloadParticles() const;

    //! Whether Celeritas core params have been created
    explicit operator bool() const { return static_cast<bool>(params_); }

    //!@}
    //!@{
    //! \name Internal use only

    using SPHitManager = std::shared_ptr<detail::HitManager>;
    using SPOffloadWriter = std::shared_ptr<detail::OffloadWriter>;
    using SPOutputRegistry = std::shared_ptr<OutputRegistry>;
    using SPState = std::shared_ptr<CoreStateInterface>;
    using SPConstGeantGeoParams = std::shared_ptr<GeantGeoParams const>;

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
 * Helper for making initialization more obvious from user code.
 */
void SharedParams::Initialize(std::string output_filename)
{
    *this = SharedParams(std::move(output_filename));
}

//---------------------------------------------------------------------------//
/*!
 * Access Celeritas data.
 *
 * This can only be called after \c Initialize.
 */
auto SharedParams::Params() const -> SPConstParams
{
    CELER_EXPECT(*this);
    return params_;
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
    CELER_ENSURE(output_reg_);
    return output_reg_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
