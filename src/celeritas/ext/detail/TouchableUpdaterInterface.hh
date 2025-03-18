//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/TouchableUpdaterInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "geocel/GeantGeoUtils.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct DetectorStepOutput;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Interface for reconstructing a touchable from HitProcessor.
 */
class TouchableUpdaterInterface
{
  public:
    // External virtual destructor
    virtual ~TouchableUpdaterInterface() = 0;

    // Update from a particular detector step
    virtual bool operator()(DetectorStepOutput const& out,
                            size_type step_index,
                            StepPoint step_point,
                            GeantTouchableBase* touchable)
        = 0;

  protected:
    TouchableUpdaterInterface() = default;
    CELER_DEFAULT_COPY_MOVE(TouchableUpdaterInterface);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
