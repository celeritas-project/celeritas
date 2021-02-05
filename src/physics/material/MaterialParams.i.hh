//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialParams.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the label for an element.
 */
const std::string& MaterialParams::id_to_label(ElementId el) const
{
    CELER_EXPECT(el < elnames_.size());
    return elnames_[el.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a material.
 */
const std::string& MaterialParams::id_to_label(MaterialId mat) const
{
    CELER_EXPECT(mat < matnames_.size());
    return matnames_[mat.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Find the material ID corresponding to a name.
 *
 * This function will have to be updated to allow for multiple MaterialDefIds
 * with same material name.
 */
MaterialId MaterialParams::find(const std::string& name) const
{
    auto iter = matname_to_id_.find(name);
    if (iter == matname_to_id_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Access material properties on the host.
 */
auto MaterialParams::host_pointers() const -> const HostRef&
{
    CELER_ENSURE(data_.host_ref);
    return data_.host_ref;
}

//---------------------------------------------------------------------------//
/*!
 * Access material properties on the device.
 */
auto MaterialParams::device_pointers() const -> const DeviceRef&
{
    CELER_ENSURE(data_.device_ref);
    return data_.device_ref;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
