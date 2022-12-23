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
namespace detail
{
class HitManager;
}
class CoreParams;
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
 * must be called to populate the Celeritas data structures (geometry,
 * physics).
 */
class SharedParams
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParams = std::shared_ptr<const CoreParams>;
    //!@}

  public:
    SharedParams() = default;
    ~SharedParams();

    // Thread-safe setup of Celeritas using Geant4 data.
    void Initialize(const SetupOptions& options);

    // Access constructed Celeritas data
    inline SPConstParams Params() const;

    //! Whether this instance is initialized
    explicit operator bool() const { return static_cast<bool>(params_); }

  private:
    //// DATA ////

    std::shared_ptr<CoreParams>         params_;
    std::shared_ptr<detail::HitManager> hit_manager_;
    std::shared_ptr<StepCollector>      step_collector_;

    //// HELPER FUNCTIONS ////

    void locked_initialize(const SetupOptions& options);
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
