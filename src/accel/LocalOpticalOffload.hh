//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOpticalOffload.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/Control.hh"
#include "celeritas/optical/gen/GeneratorData.hh"

#include "LocalOffloadInterface.hh"

class G4EventManager;

namespace celeritas
{
namespace optical
{
class CoreStateBase;
class GeneratorAction;
class Transporter;
}  // namespace optical

struct SetupOptions;
class SharedParams;

//---------------------------------------------------------------------------//
/*!
 * Manage offloading of optical distribution data to Celeritas.
 */
class LocalOpticalOffload final : public LocalOffloadInterface
{
  public:
    //!@{
    //! \name Type aliases
    using DistributionData = optical::GeneratorDistributionData;
    //!@}

  public:
    // Construct in an invalid state
    LocalOpticalOffload() = default;

    // Construct with shared (across threads) params
    LocalOpticalOffload(SetupOptions const& options, SharedParams& params);

    //!@{
    //! \name LocalOffload interface

    // Initialize with options and shared data
    void Initialize(SetupOptions const&, SharedParams&) final;

    // Set the event ID and reseed the Celeritas RNG at the start of an event
    void InitializeEvent(int) final;

    // Transport all buffered tracks to completion
    void Flush() final;

    // Clear local data and return to an invalid state
    void Finalize() final;

    // Whether the class instance is initialized
    bool Initialized() const final { return static_cast<bool>(state_); }

    // Number of buffered tracks
    size_type GetBufferSize() const final { return num_photons_; }
    //!@}

    // Offload optical distribution data to Celeritas
    void Push(DistributionData const&);

    //! Whether the class instance is initialized
    explicit operator bool() const { return this->Initialized(); }

  private:
    // Thread-local state data
    std::shared_ptr<optical::CoreStateBase> state_;

    // Transport pending optical tracks
    std::shared_ptr<optical::Transporter> transport_;

    // Buffered distributions for offloading
    std::vector<DistributionData> buffer_;

    // Action for generating optical photons from distribution data
    std::shared_ptr<optical::GeneratorAction const> generate_;

    // Accumulated number of buffered photons
    size_type num_photons_{};

    // Number of photons to buffer before offloading
    size_type auto_flush_{};

    // Current event ID or manager for obtaining it
    UniqueEventId event_id_;
    G4EventManager* event_manager_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
