//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoInterface.hh
//! Utility functions for host geometry interfaces.
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class RT>
class GeoTrackInterface;
class GeoParamsInterface;
class VolumeParams;

//---------------------------------------------------------------------------//

struct StreamableUniqueVolName
{
    GeoTrackInterface<real_type> const& geo;
    VolumeParams const& params;

    friend std::ostream& operator<<(std::ostream& os,
                                    StreamableUniqueVolName const& suvn);
};

//---------------------------------------------------------------------------//

// Get the descriptive, robust volume name based on the geo state
std::string volume_name(GeoTrackInterface<real_type> const& geo,
                        VolumeParams const& params);

// Get a robust name using impl volume params
std::string volume_name(GeoTrackInterface<real_type> const& geo,
                        GeoParamsInterface const& params);

// Get the descriptive, robust volume instance name based on the geo state
std::string volume_instance_name(GeoTrackInterface<real_type> const& geo,
                                 VolumeParams const& params);

// Get the descriptive, robust volume instance name based on the geo state
std::string unique_volume_name(GeoTrackInterface<real_type> const& geo,
                               VolumeParams const& params);

//---------------------------------------------------------------------------//
}  // namespace celeritas
