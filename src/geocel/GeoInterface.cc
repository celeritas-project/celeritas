//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoInterface.cc
//---------------------------------------------------------------------------//
#include "GeoInterface.hh"

#include "GeoParamsInterface.hh"
#include "GeoTrackInterface.hh"
#include "VolumeParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Default virtual destructor
GeoParamsInterface::~GeoParamsInterface() = default;

//---------------------------------------------------------------------------//
//! Default virtual destructor
template<class RealType>
GeoTrackInterface<RealType>::~GeoTrackInterface() = default;

#if CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
template class GeoTrackInterface<float>;
#endif
template class GeoTrackInterface<double>;

//---------------------------------------------------------------------------//

std::ostream& operator<<(std::ostream& os, StreamableUniqueVolName const& suvn)
{
    if (suvn.geo.is_outside())
    {
        return os << "[OUTSIDE]";
    }

    auto const& vi_labels = suvn.params.volume_instance_labels();
    if (vi_labels.empty())
    {
        return os;
    }

    auto vlev = suvn.geo.volume_level();
    CELER_ASSERT(vlev && vlev >= VolumeLevelId{0});

    std::vector<VolumeInstanceId> ids(vlev.get() + 1);
    suvn.geo.volume_instance_id(make_span(ids));

    os << vi_labels.at(ids[0]);
    for (auto i : range(std::size_t{1}, ids.size()))
    {
        os << '/';
        if (ids[i])
        {
            os << vi_labels.at(ids[i]);
        }
        else
        {
            os << "[INVALID]";
        }
    }
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Get the descriptive, robust volume name based on the geo state.
 */
std::string volume_name(GeoTrackInterface<real_type> const& geo,
                        VolumeParams const& params)
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }

    auto const& vol_labels = params.volume_labels();
    if (vol_labels.empty())
        return {};

    VolumeId id = geo.volume_id();
    if (!(id < vol_labels.size()))
    {
        return "[INVALID]";
    }

    return vol_labels.at(id).name;
}

//---------------------------------------------------------------------------//
/*!
 * Get the descriptive, robust impl volume name based on the geo state.
 */
std::string volume_name(GeoTrackInterface<real_type> const& geo,
                        GeoParamsInterface const& params)
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }

    auto const& vol_labels = params.impl_volumes();
    if (vol_labels.empty())
        return {};

    ImplVolumeId id = geo.impl_volume_id();
    if (!(id < vol_labels.size()))
    {
        return "[INVALID]";
    }

    return vol_labels.at(id).name;
}

//---------------------------------------------------------------------------//
/*!
 * Get the descriptive, robust volume instance name based on the geo state.
 */
std::string volume_instance_name(GeoTrackInterface<real_type> const& geo,
                                 VolumeParams const& params)
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }

    auto const& vi_labels = params.volume_instance_labels();
    if (vi_labels.empty())
        return {};

    VolumeInstanceId vi_id;
    try
    {
        vi_id = geo.volume_instance_id();
    }
    catch (celeritas::DebugError const& e)
    {
        std::ostringstream os;
        auto const& d = e.details();
        os << "<exception at " << d.file << ':' << d.line << ": "
           << d.condition << '>';
        return std::move(os).str();
    }
    if (!(vi_id < vi_labels.size()))
    {
        return "[INVALID]";
    }

    return to_string(vi_labels.at(vi_id));
}

//---------------------------------------------------------------------------//
/*!
 * Get the descriptive, robust volume instance name based on the geo state.
 */
std::string unique_volume_name(GeoTrackInterface<real_type> const& geo,
                               VolumeParams const& params)
{
    std::ostringstream os;
    os << StreamableUniqueVolName{geo, params};
    return std::move(os).str();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
