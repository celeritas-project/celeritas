//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/OpticalHitProcessorRegistry.hh
//! \brief Thread-local registration for optical hit processing.
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "celeritas/optical/DetectorData.hh"

namespace celeritas
{
namespace optical
{
class CoreParams;
}

namespace detail
{
class OpticalHitProcessor;

//---------------------------------------------------------------------------//
/*!
 * Get and set the thread-local optical hit processor pointer.
 *
 * This is used to bridge the shared-params-level DetectorAction callback
 * with the per-thread OpticalHitProcessor instance that is constructed in
 * LocalOpticalTrackOffload or LocalTransporter.
 *
 * Set to non-null during construction, reset to null in Finalize().
 * The DetectorAction callback calls through this pointer.
 */
OpticalHitProcessor*& thread_local_optical_hit_processor();

//---------------------------------------------------------------------------//
/*!
 * Build and register a thread-local OpticalHitProcessor.
 *
 * Returns the constructed processor (or nullptr if no optical detectors are
 * present). Also registers the raw pointer in the thread-local registry so
 * that the DetectorAction callback can dispatch to it.
 */
std::shared_ptr<OpticalHitProcessor>
make_optical_hit_processor(optical::CoreParams const& opt_params);

//---------------------------------------------------------------------------//
/*!
 * Unregister and destroy the thread-local OpticalHitProcessor.
 */
void reset_optical_hit_processor(std::shared_ptr<OpticalHitProcessor>&);

//---------------------------------------------------------------------------//
/*!
 * Return a DetectorAction callback that dispatches to the thread-local
 * OpticalHitProcessor, warning when the processor is unexpectedly null.
 */
std::function<void(DetectorHitsOutput const&)> make_optical_hit_callback();

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
