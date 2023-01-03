//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SharedParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Assert.hh"

namespace celeritas
{
class CoreParams;
struct SetupOptions;

//---------------------------------------------------------------------------//
/*!
 * Shared (one instance for all threads) Celeritas problem data.
 *
 * This should be instantiated on the master thread during problem setup,
 * preferably as a shared pointer. The shared pointer should be
 * passed to a thread-local \c LocalTransporter instance. At the beginning of
 * the run, after Geant4 has initialized physics data, the \c Initialize method
 * must be called first on the "master" thread to populate the Celeritas data
 * structures (geometry, physics). \c Initialize must subsequently be invoked
 * on all worker threads.
 */
class SharedParams
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParams = std::shared_ptr<const CoreParams>;
    //!@}

  public:
    // Default constructors, assignment, destructor
    SharedParams() = default;
    SharedParams(SharedParams&&)                 = default;
    SharedParams(const SharedParams&)            = default;
    SharedParams& operator=(SharedParams&&)      = default;
    SharedParams& operator=(const SharedParams&) = default;
    ~SharedParams();

    // Thread-safe setup of Celeritas using Geant4 data.
    void Initialize(const SetupOptions& options);

    // Write (shared) diagnostic output and clear shared data on master.
    void Finalize();

    // Access constructed Celeritas data
    inline SPConstParams Params() const;

    //! Whether this instance is initialized
    explicit operator bool() const { return static_cast<bool>(params_); }

  private:
    //// DATA ////

    std::shared_ptr<CoreParams> params_;
    std::string                 output_filename_;

    //// HELPER FUNCTIONS ////

    void initialize_master(const SetupOptions& options);
};

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
} // namespace celeritas
