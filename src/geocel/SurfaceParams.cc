//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceParams.cc
//---------------------------------------------------------------------------//
#include "SurfaceParams.hh"

#include <algorithm>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/io/Logger.hh"

#include "SurfaceData.hh"
#include "VolumeParams.hh"

#include "detail/SurfaceParamsBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from surface input and volume structure information.
 */
SurfaceParams::SurfaceParams(inp::Surfaces const& input,
                             VolumeParams const& volumes)
{
    CELER_EXPECT(!volumes.empty() || !input);

    // Set up temporary storage
    std::vector<detail::VolumeSurfaceData> temp_volume_surfaces;
    std::vector<Label> surface_labels;

    // Process input surfaces
    detail::SurfaceInputInserter insert_surface(
        volumes, surface_labels, temp_volume_surfaces);
    std::for_each(input.surfaces.begin(), input.surfaces.end(), insert_surface);
    labels_ = {"surfaces", std::move(surface_labels)};

    data_ = CollectionMirror<SurfaceParamsData>{[&] {
        // Convert the temporary data to device-compatible format
        HostVal<SurfaceParamsData> host_data;
        host_data.num_surfaces = labels_.size();
        // Always resize surface array, even if no surfaces are defined
        detail::VolumeSurfaceRecordBuilder build_record(
            &host_data.volume_surfaces,
            &host_data.volume_instance_ids,
            &host_data.surface_ids);
        std::for_each(temp_volume_surfaces.begin(),
                      temp_volume_surfaces.end(),
                      build_record);
        CELER_ENSURE(host_data);
        return host_data;
    }()};

    CELER_ENSURE(data_);
    CELER_ENSURE(labels_.size() == this->host_ref().num_surfaces);
}

//---------------------------------------------------------------------------//
/*!
 * Construct no surface data for when optical physics is disabled.
 */
SurfaceParams::SurfaceParams() : labels_{{"surfaces"}, {}}
{
    CELER_ENSURE(data_);
    CELER_ENSURE(labels_.size() == this->host_ref().num_surfaces);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class CollectionMirror<SurfaceParamsData>;
template class ParamsDataInterface<SurfaceParamsData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
