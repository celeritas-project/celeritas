//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/DetectorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionInterface.hh"

#include "DetectorData.hh"
#include "ScoringParams.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    DetectorAction ...;
   \endcode
 */
class DetectorAction final : public OpticalStepActionInterface,
                             public StaticConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    // Construct with ID
    explicit DetectorAction(ActionId);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::post; }

  private:
    template<MemSpace M>
    void process_hits(CoreParams const&, CoreState<M>&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
template<MemSpace M>
void DetectorAction::process_hits(CoreParams const& params,
                                  CoreState<M>& state) const
{
    DetectorHitOutput hit_results;

    // Copy hits (possibly from device) into pinned vector
    copy_hits<M>(&hit_results, state.ref().scoring);

    // Erase all hits with invalid detector ID
    hit_results.hits.erase(
        std::remove_if(hit_results.hits.begin(),
                       hit_results.hits.end(),
                       [](DetectorHit const& hit) {
                           return !static_cast<bool>(hit.detector);
                       }),
        hit_results.hits.end());

    if (!hit_results.hits.empty())
    {
        auto scoring = params.scoring();
        CELER_ASSERT(scoring);
        scoring->process_hits(make_span(hit_results.hits));
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
