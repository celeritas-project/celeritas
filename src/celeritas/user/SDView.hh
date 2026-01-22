//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/user/SDData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access sensitive detector properties.
 */
class SDView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<SDParamsData>;
    //!@}

  public:
    // Construct with shared data
    explicit inline CELER_FUNCTION SDView(ParamsRef const& params);

    // Get the detector ID of a volume
    inline DetectorId CELER_FUNCTION detector_id(ImplVolumeId iv_id);

  private:
    ParamsRef const& params_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with shared data.
 */
CELER_FUNCTION
SDView::SDView(ParamsRef const& params) : params_(params)
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the detector ID of a volume.
 */
CELER_FUNCTION DetectorId SDView::detector_id(ImplVolumeId iv_id)
{
    return params_.detectors[iv_id];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
