//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/IntegrationSingleton.hh
//---------------------------------------------------------------------------//
#pragma once

#include "accel/LocalTransporter.hh"
#include "accel/SetupOptions.hh"
#include "accel/SetupOptionsMessenger.hh"
#include "accel/SharedParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Singletons used by the Integration interfaces.
 *
 * The single singleton instance contains global data. Thread-local data is
 * managed by the \c local_transporter static class.
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

  private:
    // Only this class can construct
    IntegrationSingleton();

    //// DATA ////
    Mode mode_{Mode::size_};
    SetupOptions options_;
    SharedParams params_;
    SetupOptionsMessenger messenger_{&options_};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
