//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/IntegrationSingleton.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/sys/Stopwatch.hh"

#include "../LocalTransporter.hh"
#include "../SetupOptions.hh"
#include "../SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ScopedMpiInit;
class SetupOptionsMessenger;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Singletons used by the Integration interfaces.
 *
 * The singleton instance contains global data, including a single copy of the
 * options (which are referenced by the UI setup) and an MPI setup/teardown
 * helper.
 * Thread-local data is managed by the \c local_transporter static function.
 * Setup options are permanently referenced by the UI messenger class.
 *
 * The first call to the singleton initializes MPI if necessary, and MPI will
 * be finalized during the termination phase of the program.
 */
class IntegrationSingleton
{
  public:
    enum class Mode
    {
        disabled,
        kill_offload,
        enabled,
        size_
    };

    // Static GLOBAL shared singleton
    static IntegrationSingleton& instance();

    // Static THREAD-LOCAL Celeritas state data
    static LocalTransporter& local_transporter();

    //// ACCESSORS ////

    // Assign setup options before constructing params
    void setup_options(SetupOptions&&);

    //! Static global setup options before or after constructing params
    SetupOptions const& setup_options() const { return options_; }

    //! Whether Celeritas is enabled
    Mode mode() const { return mode_; }

    //!@{
    //! Static global Celeritas problem data
    SharedParams& shared_params() { return params_; }
    SharedParams const& shared_params() const { return params_; }
    //!@}

    //// HELPERS ////

    // Set up logging
    void initialize_logger();

    // Construct shared params on master (or single) thread
    void initialize_shared_params();

    // Construct thread-local transporter
    bool initialize_local_transporter();

    // Destroy local transporter
    void finalize_local_transporter();

    // Destroy params
    void finalize_shared_params();

    // Start the transport timer [s]
    void start_timer() { get_time_ = {}; }

    // Stop the timer and return the elapsed time [s]
    real_type stop_timer() { return get_time_(); }

  private:
    // Only this class can construct
    IntegrationSingleton();

    //// DATA ////
    Mode mode_{Mode::size_};
    SetupOptions options_;
    SharedParams params_;
    std::unique_ptr<ScopedMpiInit> scoped_mpi_;
    std::unique_ptr<SetupOptionsMessenger> messenger_;
    Stopwatch get_time_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
