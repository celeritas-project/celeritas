//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/IntegrationSingleton.hh
//---------------------------------------------------------------------------//
#pragma once

#include "accel/LocalTransporter.hh"
#include "accel/SetupOptions.hh"
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
    // Static GLOBAL shared singleton
    static IntegrationSingleton& instance();

    // Static THREAD-LOCAL Celeritas state data
    static LocalTransporter& local_transporter();

    //// ACCESSORS ////

    //! Static global setup options before constructing params
    SetupOptions& setup_options() { return options_; }

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
    void initialize_local_transporter();

    // Destroy local transporter
    void finalize_local_transporter();

    // Destroy params
    void finalize_shared_params();

  private:
    // Only this class can construct
    IntegrationSingleton() = default;

    //// DATA ////
    SetupOptions options_;
    SharedParams params_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
