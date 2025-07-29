//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeToString.cc
//---------------------------------------------------------------------------//
#include "VolumeToString.hh"

#include "VolumeParams.hh"

namespace celeritas
{
/*!
 * Convert Label to string.
 */
std::string VolumeToString::operator()(Label const& label) const
{
    return to_string(label);
}

/*!
 * Convert VolumeId to string.
 */
std::string VolumeToString::operator()(VolumeId const& id) const
{
    if (!id)
        return "<null>";

    if (params_)
    {
        return to_string(params_->volume_labels().at(id));
    }

    return "v " + std::to_string(id.get());
}

/*!
 * Convert VolumeInstanceId to string.
 */
std::string VolumeToString::operator()(VolumeInstanceId const& id) const
{
    if (!id)
        return "<null>";

    if (params_)
    {
        return to_string(params_->volume_instance_labels().at(id));
    }

    return "vi " + std::to_string(id.get());
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
