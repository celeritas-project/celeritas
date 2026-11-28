//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/OnloadProcessor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/user/OnloadData.hh"
#include "celeritas/user/OnloadSteps.hh"

#include "StepProcessor.hh"

class G4Navigator;

namespace celeritas
{
class HadronicInteractor;

namespace detail
{
class OnloadProcessor;

//! Base class providing the host/device step pipeline for onloaded steps
using OnloadProcessorBase = StepProcessor<OnloadProcessor,
                                          OnloadStepOutput,
                                          HostRef<OnloadStateData>,
                                          DeviceRef<OnloadStateData>>;

//---------------------------------------------------------------------------//
/*!
 * Transfer Celeritas onloaded steps to Geant4.
 *
 * This serves a similar purpose to the \c HitProcessor class for hit objects.
 * The asynchronous host/device pipeline is inherited from \c StepProcessor .
 */
class OnloadProcessor : public OnloadProcessorBase
{
  public:
    //! Label for profiling and tracing
    static constexpr char const* process_label = "process-onloaded-steps";

  public:
    // Construct from particles
    explicit OnloadProcessor(VecParticle const& particles);
    CELER_DEFAULT_MOVE_DELETE_COPY(OnloadProcessor);

    // Expose base-class call operators (host, device, and full output)
    using OnloadProcessorBase::operator();

    // Generate and call steps from a single onloaded step
    void operator()(OnloadStepOutput const& out, size_type i) const;

  private:
    std::shared_ptr<G4Navigator> navigator;
    std::shared_ptr<HadronicInteractor> interactor;
};

// Instantiated in OnloadProcessor.cc
extern template class StepProcessor<OnloadProcessor,
                                    OnloadStepOutput,
                                    HostRef<OnloadStateData>,
                                    DeviceRef<OnloadStateData>>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
