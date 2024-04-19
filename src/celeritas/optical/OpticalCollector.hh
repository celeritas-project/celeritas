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
struct OpticalGenStorage;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Generate scintillation and Cerenkov optical distribution data at each step.
 *
 * This builds the actions for gathering the pre-step data needed to generate
 * the optical distributions and generating the optical distributions at the
 * end of the step.
 */
class OpticalCollector
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCerenkov = std::shared_ptr<CerenkovParams const>;
    using SPConstProperties = std::shared_ptr<OpticalPropertyParams const>;
    using SPConstScintillation = std::shared_ptr<ScintillationParams const>;
    using SPGenStorage = std::shared_ptr<detail::OpticalGenStorage>;
    //!@}

    struct Input
    {
        SPConstProperties properties;
        SPConstCerenkov cerenkov;
        SPConstScintillation scintillation;
        ActionRegistry* action_registry;
        size_type buffer_capacity{};
        size_type num_streams{};

        //! True if all input is assigned and valid
        explicit operator bool() const
        {
            return (scintillation || (cerenkov && properties))
                   && action_registry && buffer_capacity > 0 && num_streams > 0;
        }
    };

  public:
    // Construct with optical params, number of streams, and action registry
    explicit OpticalCollector(Input);

    // Default destructor and move and copy
    ~OpticalCollector() = default;
    CELER_DEFAULT_COPY_MOVE(OpticalCollector);

    //! Get the distribution data
    SPGenStorage const& storage() const { return storage_; };

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
