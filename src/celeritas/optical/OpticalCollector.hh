//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/Types.hh"
#include "celeritas/optical/OpticalGenData.hh"

#include "detail/PreGenAction.hh"
#include "detail/PreGenGatherAction.hh"

namespace celeritas
{
class ActionRegistry;
class CerenkovParams;
class OpticalPropertyParams;
class ScintillationParams;

namespace detail
{
struct GenStorage;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Generate scintillation and Cerenkov optical distribution data at each step.
 */
class OpticalCollector
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCerenkov = std::shared_ptr<CerenkovParams const>;
    using SPConstProperties = std::shared_ptr<OpticalPropertyParams const>;
    using SPConstScintillation = std::shared_ptr<ScintillationParams const>;
    using SPGenStorage = std::shared_ptr<detail::GenStorage>;
    //!@}

  public:
    // Construct with optical params, number of streams, and action registry
    OpticalCollector(SPConstProperties properties,
                     SPConstCerenkov cerenkov,
                     SPConstScintillation scintillation,
                     size_type buffer_capacity,
                     size_type num_streams,
                     ActionRegistry* action_registry);

    // Default destructor and move and copy
    ~OpticalCollector() = default;
    CELER_DEFAULT_COPY_MOVE(OpticalCollector);

    //! Get the distribution data
    SPGenStorage storage() const { return storage_; };

  private:
    //// TYPES ////

    using SPPreGenAction = std::shared_ptr<detail::PreGenAction>;
    using SPGatherAction = std::shared_ptr<detail::PreGenGatherAction>;

    //// DATA ////

    SPGenStorage storage_;
    SPGatherAction gather_action_;
    SPPreGenAction pregen_action_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
