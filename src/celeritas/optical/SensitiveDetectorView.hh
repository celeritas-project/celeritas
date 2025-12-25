//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SensitiveDetectorView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/user/SDData.hh"

namespace celeritas
{
namespace optical
{

class SensitiveDetectorView
{
    using SDParamsRef = NativeCRef<SDParamsData>;

  public:
    inline CELER_FUNCTION SensitiveDetectorView(SDParamsRef const& params);

    inline auto CELER_FUNCTION detector_id(ImplVolumeId iv_id);

  private:
    SDParamsRef const& params_;
};

CELER_FUNCTION
SensitiveDetectorView::SensitiveDetectorView(SDParamsRef const& params)
    : params_(params)
{
    CELER_EXPECT(params_);
}

CELER_FUNCTION auto SensitiveDetectorView::detector_id(ImplVolumeId iv_id)
{
    return params_.detectors[iv_id];
}

}  // namespace optical
}  // namespace celeritas
