//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/DetectorView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/Types.hh"

#include "DetectorData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access sensitive detector properties.
 */
class DetectorView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<DetectorParamsData>;
    //!@}

  public:
    // Construct with shared data
    explicit inline CELER_FUNCTION DetectorView(ParamsRef const& params);

    // Get the detector ID of a volume
    inline DetectorId CELER_FUNCTION detector_id(VolumeId vol_id) const;

  private:
    ParamsRef const& params_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with shared data.
 */
CELER_FUNCTION
DetectorView::DetectorView(ParamsRef const& params) : params_(params)
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the detector ID of a volume.
 */
CELER_FUNCTION DetectorId DetectorView::detector_id(VolumeId vol_id) const
{
    return params_.detector_ids[vol_id];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
