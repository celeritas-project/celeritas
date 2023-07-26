//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
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
namespace detail
{
class HitManager;
class OffloadWriter;
}
class CoreParams;
struct Primary;
struct SetupOptions;
class StepCollector;

//---------------------------------------------------------------------------//
/*!
 * Shared (one instance for all threads) Celeritas problem data.
 *
 * This should be instantiated on the master thread during problem setup,
 * preferably as a shared pointer. The shared pointer should be
 * passed to a thread-local \c LocalTransporter instance. At the beginning of
 * the run, after Geant4 has initialized physics data, the \c Initialize method
 * must be called first on the "master" thread to populate the Celeritas data
 * structures (geometry, physics). \c InitializeWorker must subsequently be
 * invoked on all worker threads to set up thread-local data (specifically,
 * CUDA device initialization).
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

    // Construct in an uninitialized state
    SharedParams() = default;

    // Construct Celeritas using Geant4 data on the master thread.
    explicit SharedParams(SetupOptions const& options);

    // Initialize shared data on the "master" thread
    inline void Initialize(SetupOptions const& options);

    // On worker threads, set up data with thread storage duration
    static void InitializeWorker(SetupOptions const& options);

    // Write (shared) diagnostic output and clear shared data on master.
    void Finalize();

    //!@}
    //!@{
    //! \name Accessors

    // Access constructed Celeritas data
    inline SPConstParams Params() const;

    //! Get a vector of particles supported by Celeritas offloading
    inline VecG4ParticleDef const& OffloadParticles() const;

    //! Whether this instance is initialized
    explicit operator bool() const { return static_cast<bool>(params_); }

    //!@}
    //!@{
    //! \name Internal use only

    using SPHitManager = std::shared_ptr<detail::HitManager>;
    using SPOffloadWriter = std::shared_ptr<detail::OffloadWriter>;

    //! Hit manager, to be used only by LocalTransporter
    SPHitManager const& hit_manager() const { return hit_manager_; }

    //! Optional offload writer, only for use by LocalTransporter
    SPOffloadWriter const& offload_writer() const { return offload_writer_; }

    //!@}

  private:
    //// DATA ////

    std::shared_ptr<CoreParams> params_;
    SPHitManager hit_manager_;
    std::shared_ptr<StepCollector> step_collector_;
    VecG4ParticleDef particles_;
    std::string output_filename_;
    SPOffloadWriter offload_writer_;

    //// HELPER FUNCTIONS ////

    static void initialize_device(SetupOptions const& options);
    void initialize_core(SetupOptions const& options);
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
    CELER_EXPECT(*this);
    return params_;
}

//---------------------------------------------------------------------------//
/*!
 * Get a vector of particles supported by Celeritas offloading.
 *
 * This can only be called after \c Initialize.
 */
auto SharedParams::OffloadParticles() const -> VecG4ParticleDef const&
{
    CELER_EXPECT(*this);
    return particles_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
